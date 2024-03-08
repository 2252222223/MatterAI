#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
import re
import fitz, io, os
import difflib
from difflib import SequenceMatcher
import logging
from DB.DB_Manager_Tools.Knowledge_Acquisition_tools.Pdf_to_txt.Paper_to_txt import Paper_Parse

import time
import threading
import queue





logging.basicConfig(
    level=logging.ERROR, 
    format="[%(levelname)s] %(message)s")





#读取当前文件夹所有pdf
def all_file_name(file_dir, source_type=".pdf"):
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == source_type:
                z = file.split(".")
                L.append(file_dir + "/" + os.path.splitext(file)[0] + source_type)
                        #     print(L)
    return L




def Convert(source_path,goal_folder):
    start_time = time.time()
    fail_pdf = []
    if goal_folder.endswith("/"):
        pass
    else:
        goal_folder = goal_folder + "//"
    if source_path.endswith(".pdf"):
        try:
            Paper_Parse(source_path,goal_folder)
        except Exception as e:
            fail_pdf.append(source_path)
            logging.error(e, exc_info=True) # this will log the error message and the traceback
    else:
        Paper_pdfs = all_file_name(source_path)
        
        for i in Paper_pdfs:
            try:
                Paper_Parse(i, goal_folder)
            except Exception as e:
                fail_pdf.append(source_path)
                logging.error(e, exc_info=True) # this will log the error message and the traceback
    end_time = time.time()
    print(end_time-start_time)
    if len(fail_pdf)>0:
        print("转换失败文件：",fail_pdf)



# def batch_conver(pdf_list,goal_folder,q):
#     fail_pdf =[]
#     for i in pdf_list:
#         try:
#             Paper_Parse(i,goal_folder)
#         except Exception as e:
#             fail_pdf.append(source_path)
#             logging.error(e, exc_info=True) # this will log the error message and the traceback
#     q.put(fail_pdf)





# def Multi_threaded(sublists,goal_folder):
#     # 创建一个队列对象
#     q = queue.Queue()
#     # 创建一个线程列表
#     threads = []
#     # 为每个文件名创建一个线程
#     for f in sublists:
#         t = threading.Thread(target=batch_conver, args=(f,goal_folder,q))
#         threads.append(t)
#     # 启动所有线程
#     for t in threads:
#         t.start()
#     # 等待所有线程结束
#     for t in threads:
#         t.join()
#     return q.get()
#



# def Multi_Convert(source_path,goal_folder,multipath = 1):
#     start_time = time.time()
#     fail_pdf = []
#     if goal_folder.endswith("/"):
#         pass
#     else:
#         goal_folder = goal_folder + "//"
#     if source_path.endswith(".pdf"):
#         try:
#             Paper_Parse(source_path,goal_folder)
#         except Exception as e:
#             fail_pdf.append(source_path)
#             logging.error(e, exc_info=True) # this will log the error message and the traceback
#     else:
#         Paper_pdfs = all_file_name(source_path)
#         # 子列表的长度
#         n = round(len(Paper_pdfs)/multipath)
#         if n<1:
#             n=1
#         # 使用列表推导式，每隔n个元素切片
#         sublists = [Paper_pdfs[i:i+n] for i in range(0, len(Paper_pdfs), n)]
#         result = Multi_threaded(sublists,goal_folder)
#         print(result)
#
#     end_time = time.time()
#     print(end_time-start_time)
#     if len(fail_pdf)>0:
#         print("转换失败文件：",fail_pdf)








