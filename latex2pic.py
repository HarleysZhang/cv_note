import re
from urllib.parse import quote
import sys, os
current_dir = os.path.abspath(os.path.dirname(__file__)).replace('\\', '/')
sys.path.append(current_dir)

def count_file_num(self, directory, file_type = "md"):
        """count the number of json/image/.py files in directory path.
        
        Args:
            directory: root path of dat files directory.
        
        Returns:
            file_list: The file path list that require specified file type("dat").
            file_number: total file number of files that require specified file type.
        """
        if file_type == "image":
            file_types = ['jpg', 'JPG', 'png', 'PNG']
        else:
            file_types = []
            file_types.append(file_type)
        file_list = []  # specified file list
        size = 0
        try:
            # traverse subfolders in root directory
            for root, _, files in os.walk(directory):  # sub_dir is 3 tuple object
                if len(files) >= 1:
                    for sub_file in files:
                        sub_file = osp.join(root, sub_file).replace('\\', '/')
                        file_extension = osp.splitext(sub_file)[1].strip('.')  # 'png'?
                        if not file_extension in file_types:
                            continue
                        else:
                            file_list.append(sub_file)
                            size += os.path.getsize(sub_file)
        except Exception as e:
            print(e)

        file_number = len(file_list)
        print(149*"*")
        print('In 【"%s"】 directory, all 【 "%s" 】 file size is: %.3f Mb.' % (directory, file_type, size / 1024 / 1024))
        print("There are 【 %d 】 %s files." % (file_number, file_type))
        print(149*"*")

        return file_list, file_number

def show_sgl_github_md(md_path):
    assert os.path.splitext(path)[1] == '.md'
    text = open(md_path, encoding="utf-8").read()

    parts = text.split("$$")

    for i, part in enumerate(parts):
        if i & 1:
            parts[i] = f'![](http://latex.codecogs.com/gif.latex?{quote(part.strip())})'

    text_out = "\n\n".join(parts)

    lines = text_out.split('\n')
    for lid, line in enumerate(lines):
        parts = re.split(r"\$(.*?)\$", line)
        for i, part in enumerate(parts):
            if i & 1:
                parts[i] = f'![](http://latex.codecogs.com/gif.latex?{quote(part.strip())})'
        lines[lid] = ' '.join(parts)
    text_out = "\n".join(lines)

    new_file_name = os.path.basename(md_path) + '_show'
    new_file_path = osp.join(os.path.dirname(md_path), new_file_name
    with open("./深度学习/神经网络压缩算法总结new.md", "w", encoding='utf-8') as f:
        f.write(text_out)

def show_multi_github_md(md_file):
    pass

if __name__ == "__main__":
    path = "C:/Users/zhanghonggao/Documents/my_project/2020_algorithm_intern_information-master"
    if os.path.isfile(path):
        show_sgl_github_md(path)
    elif os.path.isdir(path):
        show_multi_github_md(path)
    else:
        print(FileNotFoundError)