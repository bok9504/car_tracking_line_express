import os

# 폴더에서 파일 찾아줌
def get_filepath_list_pair(_target_dir) :
    
    target_dir = os.path.normpath(_target_dir) # remove trailing separator.

    for fname in os.listdir(target_dir):
        full_dir = os.path.join(target_dir, fname)

    return full_dir

# 엔터 누르기 전까지 잠깐 멈춤
def pause():
    programPause = input("Press the <ENTER> key to continue...")
