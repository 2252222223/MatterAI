#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import os
import re
import fitz, io, os
import difflib
from difflib import SequenceMatcher
import logging
from DB.DB_Manager_Tools.Knowledge_Acquisition_tools.Pdf_to_txt.Base_tool import Base_Parse

# In[ ]:


logging.basicConfig(
    level=logging.INFO, 
    format="[%(levelname)s] %(message)s")


# In[ ]:


class Paper_Parse(Base_Parse):
    def __init__(self,pdf_path,goal_folder):
        super().__init__()
        self.goal_folder =goal_folder
        logging.info('打开文件 {}'.format(pdf_path))
        self.pdf = fitz.open(pdf_path)
        logging.info('开始获取标题')
        self.title = self.get_real_title()
        logging.info('开始获取正文内容')
        self.real_blocks_number = self.get_text()
        logging.info('开始获取doi')
        self.doi = self.get_doi()
        print(self.doi)
        logging.info('开始获取摘要')
        self.abstract = self.get_abstract()
        logging.info('保存至文本')        
        self.txt = self.pdf_to_txt()
    def get_title(self,all_font_sizes):
        doc = self.pdf
        max_font_size = max(all_font_sizes)
        for page_index, page in enumerate(doc): # 遍历每一页
            self.pdf_start_page = page_index
            cur_title = ''
            if page_index >2:
                break
            text = page.get_text("dict") # 获取页面上的文本信息
            blocks = text["blocks"] # 获取文本块列表
            for block in blocks:# 遍历每个文本块
                fonts = {}
                if block["type"] == 0 and len(block['lines']): # 如果是文字类型
                    fonts = self.get_block_font(block,fonts)
                    fonts_sizes = [float(key.split("**")[-1]) for key, value in fonts.items()]
                    if max_font_size in fonts_sizes:#找这个block里面最大的字体大小，如果和max_font_size匹配，那么进去找
#                     if max_value == max_font_size:
                        current_y1 =0
                        for b in range(len(block["lines"])):
                            for c in range(len(block["lines"][b]["spans"])):
                                if block["lines"][b]["spans"][c]["size"] == max_font_size and block["lines"][b]["dir"]== (1.0, 0.0):
                                    current_y1 = block["lines"][b]["spans"][c]["origin"][-1]
                                if block["lines"][b]["dir"]== (1.0, 0.0) and (block["lines"][b]["spans"][c]["size"]== max_font_size or current_y1-5< block["lines"][b]["spans"][c]["origin"][-1] < current_y1+5):
                                    cur_string = block["lines"][b]["spans"][c]["text"]
                                    cur_title += ' '+ cur_string
            if len(cur_title) >10:
                break
        title = cur_title.replace('  '," ")
        title = title.replace('  '," ")
    #     self.section_dict["title"] =title
        return title.lstrip()
    def get_real_title(self):
        fonts,margin = self.count_font()
#         print(fonts)
        self.margin = margin
        #获取正文的字体类型及其大小
        max_value = max(fonts.values())
        max_keys = [key for key, value in fonts.items() if value == max_value]
        self.mian_font = max_keys[0]
        all_font_sizes = list(set(list(float(i.split("**")[-1]) for i in fonts.keys())))
        for i in range(100):
            title = self.get_title(all_font_sizes)
            if 30>len(set(title.split(" ")))>5:
                break
            else:
                all_font_sizes.pop(all_font_sizes.index(max(all_font_sizes))) #如果字符串个数小于5，删除最大值

        return title
    
    def get_text(self):
        
        #去除页眉页脚的块
        self.real_blocks_number = self.obtain_research_block_number(self.pdf,self.margin,self.mian_font)
        
        #得到二级标题
        self.obtain_class_font,self.sencond_class_title,self.class_font = self.obtain_class_font(self.pdf,self.mian_font)
        a = list(self.second_title_blocks_number_and_site.keys())
#         print(self.sencond_class_title.split("**"))
        if len(self.sencond_class_title.split("**"))>1:
            if "Intr".casefold() not in " ".join(a).casefold():
                #没有introduce，但是有其它标题
                start_page = list(self.real_blocks_number.keys())[0]
                start_blocks_number = self.real_blocks_number.get(start_page)[0]
                for i in self.real_blocks_number_and_site.get(start_page):
                    if i[0]==start_blocks_number:
                        start_box = (i[1]-1,i[2]-1,0,0)
                aa={}
                aa["Introduction"] = [[start_page,start_blocks_number],[start_box]]
                aa.update(self.second_title_blocks_number_and_site)
                self.second_title_blocks_number_and_site =aa
                a = list(self.second_title_blocks_number_and_site.keys())
                self.sencond_class_title = "Introduction**" +self.sencond_class_title
        
        if self.sencond_class_title == "":
            start_key = "aaa"
            end_key = "bbb"
            self.text_0 = self.obtain_interval_text(start_key,end_key)
        else:
            for i in range(len(a)):
                setattr(self, "text_"+str(i+1), self.obtain_interval_text(a[i],a[i+1 if i+1 < len(a) else i]))
#         self.text = self.obtain_interval_text(a[0],a[1])
        return self.real_blocks_number

    def get_doi(self):
        text_list = [page.get_text() for page in self.pdf]
        source = "".join(text_list).casefold()
        doi = self.extract_substring(source, "doi:", "\n").replace(" ","").replace('/', '-').replace("\n","").replace("https:--doi.org-","").replace("http","")
        if doi == "未找到指定的起始或结束字符串。":
            doi = self.extract_substring(source, "doi.org/", "\n").replace(" ","").replace('/', '-').replace("\n","").replace("https:--doi.org-","").replace("http","")
        return doi
    
    def get_abstract(self):                                                                                                                                                                                                             
        def obtain_abstract_text(abs_font_and_size,Main_Font = False):
            # 判断start_page中哪一个块属于摘要
            abs_text = ""
            for page_index, page in enumerate(self.pdf): # 遍历每一页
                if page_index == self.pdf_start_page:
                    text = page.get_text("dict") # 获取页面上的文本信息
                    blocks = text["blocks"] # 获取文本块列表
                    for block in blocks: # 遍历每个文本块
                        a = list(self.second_title_blocks_number_and_site.keys())
                        if Main_Font is True:
                            if len(a) >1 and block["number"] > self.second_title_blocks_number_and_site.get(a[0])[0][1]:
                                break
                        fonts = {}
                        if block["type"] == 0 and len(block['lines']): # 如果是文字类型
                            fonts = self.get_block_font(block,fonts)
                            max_value = max(fonts.values())
                            max_keys = [key for key, value in fonts.items() if value == max_value]
                            if len(a)>0:
                                if max_keys == abs_font_and_size:
                                    abstract_number = block["number"]
    #                                 print(abstract_number)
                                    abstract = self.get_block_text(block)
                                    abs_text += abstract + " "                                   
                            else:
                                if max_keys == abs_font_and_size:
                                    abstract_number = block["number"]
    #                                 print(abstract_number)
                                    abstract = self.get_block_text(block)
                                    abs_text += abstract + " "
            return abs_text
        
        #摘要应该和标题处于同一页，所以首先解析标题页的所有字体类型及其数量
        first_page_fornt_and_size,_ = self.count_font(custom_page = self.pdf_start_page)
#         print(first_page_fornt_and_size)
        if self.mian_font in first_page_fornt_and_size.keys():
            del first_page_fornt_and_size[self.mian_font]

        #得到摘要的字体，大小
        abs_font_and_size = [key for key, value in first_page_fornt_and_size.items() if value == max(list(first_page_fornt_and_size.values()))]
#         print(abs_font_and_size)
        
        abs_text = obtain_abstract_text(abs_font_and_size)
        if len(abs_text)>400:
            pass
        else:
            print("摘要字体和正文相同")
            
            abs_text = obtain_abstract_text([self.mian_font],Main_Font = True)
        #获取摘要
        return abs_text
    
    def pdf_to_txt(self):
        def text_split(string):
            string = string.replace("  "," ")
            aa = string.split("**")
            new_string = ""
            for i in range(len(aa)):
                my_seq = SequenceMatcher(a = self.abstract, b = aa[i])
                if len(aa[i])==0 or my_seq.ratio() > 0.7:
                    pass
                elif i >0 and len(aa[i-1])>0 and (list(aa[i-1])[-1] == "." or list(aa[i-1])[-1] =="?" or list(aa[i-1])[-1] =="]" or list(aa[i-1])[-1] ==" " or list(aa[i-1])[-1].isdigit()):
                    new_string += "\n\n" + aa[i].strip()
                else:
                    new_string += " " + aa[i]
            return new_string.strip().replace("  ", " ")
        #去除重复项
        #使用vars()函数获取所有属性
        attributes = vars(self)
        #使用for循环遍历属性
        for key, value in attributes.items():
            #使用startswith()方法判断属性是否以self.text开头
            if key.startswith("text"):
                if list(key)[-2].isdigit():
                    num = list(key)[-2] + list(key)[-1]
                    num = int(num)
                else:
                    num = int(list(key)[-1])
                if num==0:
                    pass
                else:
                    second_title = self.sencond_class_title.split("**")[num-1]
                    #查找重复项
                    if num==1:
                        if second_title in self.abstract:
                            overlap_text = self.extract_substring(self.abstract,second_title)
                            if overlap_text not in self.text_1:
                                self.text_1 = overlap_text + " " + self.text_1
                            self.abstract = self.abstract[:self.abstract.find(second_title)]
                    else:
                        if second_title in getattr(self,"text_" + str(num-1)):
                            overlap_text = self.extract_substring(getattr(self,"text_" + str(num-1)),second_title)
                            new_text = overlap_text + " " + getattr(self,"text_" + str(num))
                            setattr(self,"text_" + str(num),new_text)
                            new_before_text = getattr(self,"text_" + str(num-1))[:getattr(self,"text_" + str(num-1)).find(second_title)]
                            setattr(self,"text_" + str(num-1),new_before_text)
        text = "Title:"
        text += self.title + "\n\n" + "Abstract:" + self.abstract.replace("  "," ").replace("**","") + "\n\n"

        attributes = vars(self)
        for key, value in attributes.items():
            #使用startswith()方法判断属性是否以self.text开头
            if key.startswith("text"):
                if list(key)[-2].isdigit():
                    num = list(key)[-2] + list(key)[-1]
                    num = int(num)
                else:
                    num = int(list(key)[-1])
                if num==0:
                    text += "Main Text:" + "\n" + text_split(value.replace("\xa0 ", " ")) + "\n\n"                    
                else:
                    second_title = self.sencond_class_title.split("**")[num-1]
                    text += second_title + "\n" + text_split(value.replace("\xa0 ", " ")) + "\n\n"
        
        filename = self.goal_folder + self.doi + ".txt"
        text = text.replace("  "," ")
        text = text.replace("- ", "-")
#         text = text.encode('utf-8', 'replace').decode('utf-8')
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(text)     
        return text

