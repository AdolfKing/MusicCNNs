#coding=utf8
import os
import sys


if __name__ == '__main__':
    #path = sys.argv[1]
    #print(path)

    all_genre = os.listdir('.')
    for d in all_genre:
        if os.path.isdir(d):
            all_files = os.listdir(d)
            all_files.sort()
            print('Deleting %s..'%d)
            for i,f in enumerate(all_files):
                path = d+'/'+f
                if os.path.isfile(path):
                    if i>100:
                        os.remove(path)
