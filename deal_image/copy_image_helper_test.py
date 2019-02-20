# coding: utf-8

'''
this script aims to copy image to destination directory .
Author: chenglong Yi
date: 2019-02-20

'''

from copy_image_helper import CopyImageHelper


def test_copyfile():
    helper =  CopyImageHelper()
    src_path = "/mnt/hgfs/share_vm/temp_data/Jiankong_data/test"
    dst_path = "/mnt/hgfs/share_vm/temp_data/validation_data/smoke1/"
    helper.copy_each_file(src_path, dst_path)

    print("Done ...")




if __name__ == "__main__":
    test_copyfile()
