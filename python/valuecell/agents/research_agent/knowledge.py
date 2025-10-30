from pathlib import Path
from typing import Optional

import pyarrow as pa
from agno.knowledge.chunking.markdown import MarkdownChunking
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.reader.markdown_reader import MarkdownReader
from agno.knowledge.reader.pdf_reader import PDFReader
from loguru import logger

from .vdb import vector_db


def _ensure_vector_schema() -> None:
    """确保 LanceDB 表的向量维度与当前嵌入模型一致。"""

    table = getattr(vector_db, "table", None)
    expected_dim = getattr(vector_db, "dimensions", None)
    vector_col = getattr(vector_db, "_vector_col", None)

    if table is None or expected_dim is None or vector_col is None:
        return

    try:
        field = table.schema.field(vector_col)
    except KeyError:
        logger.warning("LanceDB 表缺少列 %s，将重建表。", vector_col)
        field = None

    current_dim: Optional[int] = None
    if field is not None:
        field_type = field.type
        if pa.types.is_fixed_size_list(field_type):
            current_dim = field_type.list_size
        elif pa.types.is_list(field_type):
            current_dim = None  # 动态列表，不需要重建
    else:
        current_dim = -1  # 强制触发重建

    if current_dim is not None and current_dim != expected_dim:
        logger.warning(
            "检测到 LanceDB 向量维度不匹配（当前=%s, 期望=%s），将覆盖重建表。",
            current_dim,
            expected_dim,
        )
        schema = vector_db._base_schema()  # pylint: disable=protected-access
        vector_db.connection.create_table(
            name=vector_db.table_name,
            schema=schema,
            mode="overwrite",
        )
        vector_db.table = vector_db.connection.open_table(name=vector_db.table_name)
        logger.info("LanceDB 表 %s 已重建。", vector_db.table_name)


_ensure_vector_schema()

knowledge = Knowledge(
    vector_db=vector_db,
    max_results=10,
)
md_reader = MarkdownReader(chunking_strategy=MarkdownChunking())
pdf_reader = PDFReader(chunking_strategy=MarkdownChunking())


async def insert_md_file_to_knowledge(
    name: str, path: Path, metadata: Optional[dict] = None
):
    await knowledge.add_content_async(
        name=name,
        path=path,
        metadata=metadata,
        reader=md_reader,
    )


async def insert_pdf_file_to_knowledge(url: str, metadata: Optional[dict] = None):
    await knowledge.add_content_async(
        url=url,
        metadata=metadata,
        reader=pdf_reader,
    )
