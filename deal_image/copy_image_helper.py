# coding: utf-8

'''
this script aims to copy image to destination directory .
Author: chenglong Yi
date: 2019-02-20

'''

import os
from shutil import copyfile


class CopyImageHelper(object):
    def __init__(self):
        self.total_count = 0
        
    
    def get_cur_path(self, file_path_name):
    	(filepath,tempfilename) = os.path.split(file_path_name)
        (shotname,extension) = os.path.splitext(tempfilename)
        return (filepath, shotname, extension)


    def copy_each_file(self, filepath, dst_dir_img):
        pathDir = os.listdir(filepath)      #获取当前路径下的文件名，返回List
        for s in pathDir:
            newDir = os.path.join(filepath,s)     #将文件命加入到当前文件路径后面
            if os.path.isfile(newDir):              #如果是文件
                if os.path.splitext(newDir)[1]==".jpg":  #判断是否是 .jpg
                    self.total_count = self.total_count + 1
                    (filepath, shotname, extension) = self.get_cur_path(newDir)
                    dst_file = dst_dir_img + shotname + extension
                    print("idx:{0}, srcfile:{1} ---> desfile:{2}".format(self.total_count, newDir, dst_file))
                    copyfile(newDir, dst_file)
            else:
                self.copy_each_file(newDir, dst_dir_img)                #如果不是文件，递归这个文件夹的路径

