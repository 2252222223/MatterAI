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


# In[ ]:


class Base_Parse:
    def __init__(self):
        self.class_title_characteristic =["1.Introduction","Introduction","INTRODUCTION","RESULTS AND DISCUSSION","Experimentals","Results and discussion",
                                         "Results","Discussion","Conclusion"]
    #获取某一页，或者某个文件的字体大小-数量，以及页边距
    def count_font(self,custom_page = None):
        fonts = {}
        margin = {}
        doc = self.pdf # 打开pdf文件
        for page_index, page in enumerate(doc): # 遍历每一页
            if custom_page is None:
                text = page.get_text("dict") # 获取页面上的文本信息
                blocks = text["blocks"] # 获取文本块列表
                x1,y1,x2,y2 = 10000,10000,0,0
                for block in blocks: # 遍历每个文本块
                    x1_,y1_,x2_,y2_ = block["bbox"]
                    x1 = min(x1_, x1)
                    y1 = min(y1_, y1)
                    x2 = max(x2_, x2)
                    y2 = max(y2_, y2)
                    if block["type"] == 0 and len(block['lines']) and page_index < (len(self.pdf)/1.5): # 如果是文字类型
                        fonts = self.get_block_font(block,fonts)
                margin[page_index] = (x1,y1,x2,y2)
            elif page_index == custom_page:
                text = page.get_text("dict") # 获取页面上的文本信息
                blocks = text["blocks"] # 获取文本块列表
                x1,y1,x2,y2 = 10000,10000,0,0
                for block in blocks: # 遍历每个文本块
                    x1_,y1_,x2_,y2_ = block["bbox"]
                    x1 = min(x1_, x1)
                    y1 = min(y1_, y1)
                    x2 = max(x2_, x2)
                    y2 = max(y2_, y2)
                    if block["type"] == 0 and len(block['lines']): # 如果是文字类型
                        fonts = self.get_block_font(block,fonts)
                margin[page_index] = (x1,y1,x2,y2)
            else:
                pass
        return fonts,margin

    def get_block_text(self,block):
        cur_text =""
        if block["type"] == 0 and len(block['lines']): # 如果是文字类型
            bbox_min_x = 10000 #block最小的x坐标
            
            for b in range(len(block["lines"])):
                if block["lines"][b]["dir"]== (1.0, 0.0):
                    for c in range(len(block["lines"][b]["spans"])):
                        current_x1,current_y1 = block["lines"][b]["spans"][c]["origin"][0],block["lines"][b]["spans"][c]["origin"][1]
                        cur_string = block["lines"][b]["spans"][c]["text"]
                        if current_x1 < bbox_min_x:
                            bbox_min_x = current_x1
                        
                        #判断是否换段落了
                        if c > 0 or b > 0:
                            if current_y1 > (before_y1+5) and current_x1 > (bbox_min_x + 7) and len(list(cur_string)) >0 and (list(cur_string)[0].istitle() or list(cur_string)[0].isdigit() or list(cur_string)[0]==" "):
                                cur_text = cur_text + "**"
#                         print(b,c,cur_string)
                        if b==0 and c==0 and len(list(cur_string)) >0 and (list(cur_string)[0].istitle() or list(cur_string)[0].isdigit()): #每个bulock第一行判断是否为段落开头 
                            cur_text = "**" + cur_text
                        if cur_text == '':
                            cur_text += cur_string
                        else:
                            if (c > 0 or b > 0) and current_y1 > (before_y1+5):
                                cur_text += ' '+ cur_string
                            else:
                                cur_text += cur_string
                        before_y1 = current_y1
#         cur_text += "**"
        return cur_text

    def get_block_font(self,block,fonts):
        if block["type"] == 0 and len(block['lines']): # 如果是文字类型
            if len(block["lines"][0]["spans"]):
                for i in range(len(block["lines"])):
                    for b in range(len(block["lines"][i]["spans"])):
                        font = block["lines"][i]["spans"][b]["font"]
                        font_size = str(block["lines"][i]["spans"][b]["size"]) # 获取第一行第一段文字的字体大小 
                        text = block["lines"][i]["spans"][b]["text"] #获取当前文字
                        if font + "**" + font_size in fonts:
                            fonts[font + "**" + font_size] = len(text) + fonts.get(font + "**" + font_size)
                        else:
                            fonts[font + "**" + font_size] = len(text)
        return fonts
    
    def bulk_parse(self,block,margin,mian_font,page_index):
        fonts = {}
        x1_,y1_,x2_,y2_ = block["bbox"]
        #如果bulk处于页面上部或者下部，需要判断是否属于正文
        if y1_ < ((margin[3]-margin[1])* (0.3 if page_index == self.pdf_start_page else 0.03) + margin[1]) or y2_ > (margin[3]-(margin[3]-margin[1])*0.3):
#             print("可能为页眉or页脚")
            if block["type"] == 0 and len(block['lines']): # 如果是文字类型
                fonts = self.get_block_font(block,fonts)
                max_value = max(fonts.values())
                max_keys = [key for key, value in fonts.items() if value == max_value]
#                 print(block["number"],max_keys[0],mian_font)
                if max_keys[0] == mian_font or max_value > 1000:
                    return False
                else:
                    return True 
        else:
            return False
        
    def obtain_research_block_number(self,pdf,margin,mian_font):
        #排block的先后排序
        def rank_block(page_index,block_number_and_site):
            if len(block_number_and_site) ==0:
                return []
            else:
                left_index = np.where(np.array(block_number_and_site)[:,1] < (self.margin.get(page_index)[2]-self.margin.get(page_index)[0])*0.45 + self.margin.get(page_index)[0]) #block在左边的索引
                rank_left_index = np.argsort(np.array(block_number_and_site)[left_index][:,-1])
                left_block_number = np.array(block_number_and_site)[left_index][rank_left_index][:,:1]
                right_index = np.where(np.array(block_number_and_site)[:,1] > (self.margin.get(page_index)[2]-self.margin.get(page_index)[0])*0.45 + self.margin.get(page_index)[0]) #block在右边的索引
                rank_right_index = np.argsort(np.array(block_number_and_site)[right_index][:,-1])
                right_block_number = np.array(block_number_and_site)[right_index][rank_right_index][:,:1]
                rank_block_number = list(np.concatenate((left_block_number,right_block_number),axis = 0).reshape(-1))
                return rank_block_number
        """获取pdf每一页的研究内容（除去页眉页脚）"""
        self.real_blocks_number = {}
        self.real_blocks_number_and_site = {}
        for page_index, page in enumerate(pdf): # 遍历每一页

#             block_number = []
            block_number_and_site = []
            text = page.get_text("dict") # 获取页面上的文本信息
            blocks = text["blocks"] # 获取文本块列表
            kkk = 0
            for block in blocks: # 遍历每个文本块

                if block["type"] == 0 and len(block['lines']): # 如果是文字类型
                    if self.bulk_parse(block,margin[page_index],mian_font,page_index):
#                         print("跳过",page_index,block["number"])
                        pass
                    else:
#                         print(block["number"],"开始判断是否为图注")
                        #进一步判断是否属于图注
                        text = block["lines"][0]["spans"][0]["text"]
                        #文字为Fig 并且上一个block为图片
                        if len(list(text)) > 2 and list(text)[0] =="F" and list(text)[1] == "i" and block["number"] >0 and blocks[block["number"]-1]["type"]==1:
                            pass
                        else:
                            block_number_and_site.append([kkk,block["bbox"][0],block["bbox"][1]])
                kkk +=1
#             print(block_number_and_site)
            rank_block_number = rank_block(page_index,block_number_and_site)
            self.real_blocks_number[page_index] = rank_block_number
            self.real_blocks_number_and_site[page_index] = block_number_and_site
            
        return self.real_blocks_number
    
    #根据字体类型和大小，寻找文档指定内容
    def obtain_custom_text(self,pdf,custom_font_and_size):
        def find_keys(dictionary, value):

            keys = []
            for key, val in dictionary.items():
                #需要满足同一个块，且bbox差距小于40
                if val[0][0] == value[0][0] and val[0][1] == value[0][1] and value[1][0][1]-40 < val[1][0][1] <value[1][0][1]+40:
                    keys.append(key)
            return keys
        
        new_dict = {}
        custom_font,custom_font_size = custom_font_and_size.split("**")
        custom_font_size = float(custom_font_size)
        cur_text = ''
        for page_index, page in enumerate(pdf): # 遍历每一页
            text = page.get_text("dict") # 获取页面上的文本信息
            blocks = text["blocks"] # 获取文本块列表
            for block in blocks: # 遍历每个文本块
                if block["type"] == 0 and len(block['lines']): # 如果是文字类型
                    for b in range(len(block["lines"])):
                        for c in range(len(block["lines"][b]["spans"])):
                            if block["lines"][b]["spans"][c]["size"] == custom_font_size and block["lines"][b]["spans"][c]["font"] == custom_font:
                                cur_string = block["lines"][b]["spans"][c]["text"]

                                new_dict[cur_string] = [[page_index,block["number"]],[block["lines"][b]["spans"][c]["bbox"]]]
                                
        if " " in new_dict.keys():
            del new_dict[" "]
        #如果block相同，并且bbox相差不大，则合并
        self.second_title_blocks_number_and_site = {}
        for i in new_dict.values():
            keys = find_keys(new_dict, i)
            
            self.second_title_blocks_number_and_site[" ".join(keys)] = i        
                              
        return "**".join(list(self.second_title_blocks_number_and_site.keys()))
    
    def extract_substring(self,source, abs_start, abs_end = None):#提取指定字符串中间位置
        start_idx = source.find(abs_start)
    #         print("start_idx:{}".format(start_idx))
        if abs_end is None:
            end_idx = -1
        else:
            end_idx1 = source.find(abs_end, start_idx)
            if end_idx1==-1:
                end_idx1 = 1000000000
#             print("end_idx1 - start_idx:",end_idx1 - start_idx)
            if end_idx1 - start_idx <14:

                end_idx1 = source.find(abs_end,end_idx1+1)  
            end_idx2 = source.find(" ", start_idx)
#             print(end_idx2)
            if end_idx2 - start_idx <10:
                end_idx2 = source.find(" ",end_idx2+1)
            end_idx = end_idx1 if end_idx1 < end_idx2 else end_idx2
#             print(start_idx,end_idx)
        if start_idx == -1:
            return "未找到指定的起始或结束字符串。"
        return source[start_idx + len(abs_start): end_idx]
    
    
    def obtain_class_font(self,pdf,mian_font):
        """获取字体所属级别，明确一级标题，二级标题，"""
        class_font = {}
        for page_index, page in enumerate(pdf): # 遍历每一页
            text = page.get_text("dict") # 获取页面上的文本信息
            blocks = text["blocks"] # 获取文本块列表
            for block in blocks: # 遍历每个文本块
                if block["number"] in self.real_blocks_number.get(page_index):#该块是否属于正文
                    class_font = self.get_block_font(block,class_font)
        #得到去除页眉，页脚，标题后大于等于正文字体大小的所有字体
        font_clean = {k: v for k, v in class_font.items() if float(k.rsplit("**", 1)[-1].casefold()) >= float(mian_font.rsplit("**", 1)[-1].casefold())}

        #去掉正文字体
        del font_clean[mian_font]
        
        #取每个字体的所有文本，判断是否可能为二级标题
        sencond_class_title = ""
        for custom_font_and_size in font_clean.keys():
            cur_text = self.obtain_custom_text(pdf,custom_font_and_size)

            for feature in self.class_title_characteristic:
                if feature in cur_text:
                    sencond_class_font_and_size = custom_font_and_size
                    sencond_class_title = cur_text
                    break           
            if len(sencond_class_title) >3:
                break     
        return font_clean,sencond_class_title,class_font
    
    #根据区间（起始页，bulcks），寻找文档指定内容
    def obtain_interval_text(self,start_key,end_key):
        #特殊情况，无二级标题
        if start_key=="aaa" and end_key =="bbb":
            start_page,end_page = list(self.real_blocks_number.keys())[0],list(self.real_blocks_number.keys())[-1]
            start_blocks_number,end_blocks_number = self.real_blocks_number.get(start_page)[0],self.real_blocks_number.get(end_page)[-1] if len(self.real_blocks_number.get(end_page))>0 else 0  
                                                                    
            for i in self.real_blocks_number_and_site.get(start_page):
                if i[0]==start_blocks_number:
                    start_box = (i[1]-1,i[2]-1,0,0)
            if len(self.real_blocks_number_and_site.get(end_page))>0:      
                for i in self.real_blocks_number_and_site.get(end_page):
                    if i[0]==end_blocks_number:
                        end_box = (i[1]-1,i[2]-1,0,0)
            else:
                end_box = (0,0,0,0)
                
        else:
            start_page,start_blocks_number,start_box = self.second_title_blocks_number_and_site.get(start_key)[0][0],self.second_title_blocks_number_and_site.get(start_key)[0][1],self.second_title_blocks_number_and_site.get(start_key)[1][0]

            if start_key==end_key:
                end_page,end_blocks_number,end_box = 100000,100,(0,0,0,0)
            else:
                end_page,end_blocks_number,end_box = self.second_title_blocks_number_and_site.get(end_key)[0][0],self.second_title_blocks_number_and_site.get(end_key)[0][1],self.second_title_blocks_number_and_site.get(end_key)[1][0]

        #首先判断二级标题位于整个页面的方位
#         print(start_box)
        x1_start,y1_start,x2_start,y2_start = start_box
        
        x1_end,y1_end,x2_end,y2_end = end_box
        
        x1,y1,x2,y2 = self.margin.get(start_page)
        
        
        if (x1_start-x1)/(x2-x1) >0.4:
            Start_Right = True
        else:
            Start_Right = False
            
        if start_key==end_key:
            pass
        else:
            x_1,y_1,x_2,y_2 = self.margin.get(end_page)
        
        if (x1_end-x1)/(x2-x1) >0.4:
            End_Right = True
        else:
            End_Right = False

        cur_text = ""
        for page_index, page in enumerate(self.pdf): # 遍历每一页
            if end_page >= page_index >= start_page:
                text = page.get_text("dict") # 获取页面上的文本信息
                blocks = text["blocks"] # 获取文本块列表
                for number in self.real_blocks_number.get(page_index): # 遍历每个文本块
                    block = blocks[int(number)]
#                     if block["number"] in self.real_blocks_number.get(page_index): #每个文本块要属于正文                       
                    if page_index == start_page == end_page:
                        if Start_Right:#都在右边
                            if y1_end > block["bbox"][1] >= y1_start and (block["bbox"][0]-x1)/(x2-x1) > 0.4: #在标题的blocks下面才取文字
                                cur_string = self.get_block_text(block)
                                cur_text += cur_string

                        elif Start_Right is False and End_Right is False:#都在左边
                            if y1_end > block["bbox"][1] >= y1_start and (block["bbox"][0]-x1)/(x2-x1) < 0.4: #在标题的blocks下面才取文字
                                cur_string = self.get_block_text(block)
                                cur_text += cur_string                                
                        else:#一左一右
                            if block["bbox"][1] >= y1_start and (block["bbox"][0]-x1)/(x2-x1) < 0.4: #在标题的blocks下面才取文字
                                cur_string = self.get_block_text(block)
                                cur_text += cur_string                                      
                            elif block["bbox"][1] < y1_end and (block["bbox"][0]-x1)/(x2-x1) > 0.4:
                                cur_string = self.get_block_text(block)
                                cur_text += cur_string

                    elif page_index == start_page:
#                             print("here")
                        if Start_Right:#当二级标题处于右边时
                            if block["bbox"][1] >= y1_start and (block["bbox"][0]-x1)/(x2-x1) > 0.4: #在标题的blocks下面才取文字
                                cur_string = self.get_block_text(block)
                                cur_text += cur_string
                        else:#当二级标题处于左边时
#                             print(block["number"])
                            if block["bbox"][1] >= y1_start and (block["bbox"][0]-x1)/(x2-x1) < 0.4: #在标题的blocks下面才取文字
#                                 print("在标题下面，取")
                                cur_string = self.get_block_text(block)
                                cur_text += cur_string
                            if (block["bbox"][0]-x1)/(x2-x1) > 0.45:
#                                 print("在标题右边，取")
                                cur_string = self.get_block_text(block)
                                cur_text += cur_string

                    elif page_index == end_page:
                        
                        if End_Right:#当二级标题处于右边时
                            if block["bbox"][1] < y1_end and (block["bbox"][0]-x_1)/(x_2-x_1) > 0.45: #在标题的blocks下面才取文字
                                cur_string = self.get_block_text(block)
                                cur_text += cur_string
                            elif (block["bbox"][0]-x_1)/(x_2-x_1) < 0.4:
                                cur_string = self.get_block_text(block)
                                cur_text += cur_string
                            else:
                                pass
                        else:#当二级标题处于左边时
                            if block["bbox"][1] < y1_end and (block["bbox"][0]- x_1)/(x_2 - x_1) < 0.4: #在标题的bulcks下面才取文字
                                cur_string = self.get_block_text(block)
                                cur_text += cur_string
                    else:
                        cur_string = self.get_block_text(block)
                        cur_text += cur_string
        return cur_text

