from pathlib import Path


def get_vector_list():
    path = "D:\OneDrive - mails.ucas.ac.cn\Code\未分类\MatterAI-0816-only-test\DB\DB_Manager_Tools\Knowledge_Acquisition_tools\Domain Vector Library"
    subdirs = []  # 创建一个空列表
    p = Path(path)  # 创建一个Path对象，表示root目录
    for child in p.iterdir():  # 遍历root目录下的所有子目录和文件
        if child.is_dir():  # 如果是子目录
            subdirs.append(child.name)  # 将子目录名添加到列表中
    available_vector = ",".join(subdirs).replace(" vector_store", "")
    return available_vector


available_vectors = get_vector_list()

