import cv2
import unittest
import os
from pathlib import Path as P
from algorithm import Algorithm


class Test(unittest.TestCase):
    def test(self):
        imgs_path = '/data00/xyc_temp/mydata/xfl_audit/pos'
        imgs_path_list = [os.path.join(imgs_path, img) for img in os.listdir(imgs_path)]    
        imgs_path_list.sort()
        algorithm = Algorithm()
        #outfile = open('cloth_save_neg.txt', 'a', encoding='utf-8')
        for img_path in imgs_path_list:
            im = cv2.imread(img_path)
            im = im[:,:,::-1]
            res = algorithm.process([im], {'method':'all', 'thres_face':0.48, 'thres_cloth':0.8})
            
            #outfile.write(img_path.split('/')[-1] + ' ' + str(res) + '\n')
            print(img_path, res)
        #outfile.close()


if __name__ == "__main__":
    unittest.main()
