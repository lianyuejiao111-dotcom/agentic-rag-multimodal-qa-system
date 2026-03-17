import os
import io
import zipfile
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_deepseek import ChatDeepSeek
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend

from agentic_rag.file_processor import FileProcessor
from agentic_rag.prompt.agentic_report_prompt import SYSTEM_PROMPT
from agentic_rag.tools.base_rag import ask_base_rag, search_base_rag
from agentic_rag.database.history_repository import log_history, get_history
from agentic_rag.database.history_tables import ensure_history_table
from agentic_rag.database.db import engine, Base
from agentic_rag.qdrant_manager import QDRANT_MANAGER
from agentic_rag.database import models  # noqa: F401  # 注册 ORM 模型，便于 create_all
load_dotenv()
qdrant_manager = QDRANT_MANAGER()
# Initialize DeepSeek model
deepseek_model = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

# from sqlalchemy.orm import sessionmaker
# from sqlalchemy import create_engine
# from config import DATABASE_URL
#
# engine = create_engine(
#     DATABASE_URL,
#     connect_args={'client_encoding': 'utf8'}
# )
# SessionLocal = sessionmaker(bind=engine)
#
# qwen_embedding = LLMClient(provider="qwen-cn",model="text-embedding-v4")
# qdrant_manager = QDRANT_MANAGER()

app = FastAPI()


@app.on_event("startup")
async def on_startup() -> None:
    """
    服务启动时，自动检查并创建基础表和各接口的历史表。
    """
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    # 为已知接口提前创建历史表；新接口可在这里补充
    for iface in ("rag_base", "rag_agentic"):
        await ensure_history_table(iface)


@app.post("/upload/zip")
async def upload_zip(file: UploadFile = File(...)):

    try:
        file_processor = FileProcessor(qdrant_manager=qdrant_manager)

        # 读取 ZIP 内容
        zip_content = await file.read()
        print(f"[INFO] ZIP 大小: {len(zip_content)} bytes")

        # 打开 ZIP
        with zipfile.ZipFile(io.BytesIO(zip_content)) as zip_file:

            # ------------- ① 中文文件名解码（你要求保留的方式） -------------
            decoded_files = []
            decode_map = {}  # 原始 → 解码后文件名映射表

            for raw_name in zip_file.namelist():
                decoded_name = raw_name
                try:
                    decoded_name = raw_name.encode("cp437").decode("gbk")
                except:
                    try:
                        decoded_name = raw_name.encode("cp437").decode("gb2312")
                    except:
                        pass

                decoded_files.append(decoded_name)
                decode_map[decoded_name] = raw_name  # 保留映射用于读取文件内容

            print(f"[INFO] ZIP 文件总数: {len(decoded_files)}")

            # ------------- ② 筛选处理文件 -------------
            doc_exts = ('.pdf', '.docx', '.pptx', '.doc')
            img_exts = ('.png', '.jpg', '.jpeg', '.bmp', '.gif', '.webp', '.tiff', '.tif')

            target_files = [
                f for f in decoded_files if f.lower().endswith(doc_exts + img_exts)
            ]

            if not target_files:
                raise HTTPException(status_code=400, detail="未找到可处理的文档或图片文件")

            results = []

            # ------------- ③ 处理每个文件 -------------
            for decoded_name in target_files:
                try:
                    raw_name = decode_map[decoded_name]
                    file_bytes = zip_file.read(raw_name)

                    ext = os.path.splitext(decoded_name)[1].lower()

                    # 处理文档或图片
                    if ext in img_exts:
                        result = file_processor.process_image_file(file_bytes, decoded_name)
                    else:
                        result = file_processor.process_file_content(file_bytes, decoded_name)

                    results.append({"filename": decoded_name, "status": "success"})

                except Exception as e:
                    results.append({
                        "filename": decoded_name,
                        "status": "failed",
                        "error": str(e)
                    })

            file_processor.cleanup()

            return {
                "message": "文件处理完成",
                "results": results
            }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

@app.post("/rag/base")
async def base_rag(message: str, request: Request):
    ai_response = ask_base_rag(message)
    await log_history(
        "rag_base",
        request_text=message,
        response_text=ai_response,
        user_id="3",
        meta={"endpoint": "/rag/base"},
    )
    return ai_response


@app.get("/rag/base/history")
async def base_rag_history(
    limit: int = 20, offset: int = 0, user_id: Optional[str] = None
):
    return await get_history("rag_base", limit=limit, offset=offset, user_id=user_id)

@app.post("/rag/agentic")
async def agentic_rag(message: str, request: Request):
    agentic_rag_agent = create_deep_agent(
        model=deepseek_model,
        tools=[search_base_rag],
        system_prompt=SYSTEM_PROMPT,
        backend=FilesystemBackend(root_dir="./report_output", virtual_mode=True),
        debug=True
    )
    result = agentic_rag_agent.invoke({"messages": [
        {"role": "user", "content": message}]})
    content = result["messages"][1].content
    await log_history(
        "rag_agentic",
        request_text=message,
        response_text=content,
        user_id="2",
        meta={"endpoint": "/rag/agentic"},
    )
    return content


@app.get("/rag/agentic/history")
async def agentic_rag_history(
    limit: int = 20, offset: int = 0, user_id: Optional[str] = None
):
    return await get_history("rag_agentic", limit=limit, offset=offset, user_id=user_id)

# 启动提示
@app.get("/")
def root():
    return {"message": "FastAPI ZIP Upload Server is running!"}
