### code from the notebook pycocoEvalDemo.ipynb

from pycocotools.coco import COCO
import cocoeval_modif
import numpy as np
import pandas as pd
from os.path import join

import logging
logging.basicConfig(format='%(levelname)s: %(filename)s L.%(lineno)d  -  %(message)s',
    level=logging.INFO)

### Validation File:
# validFile = '/home/javier/CLEAR_IMAGE_AI/coco_validation_jsons/instances_val2014.json'
# validFile = '/home/javier/CLEAR_IMAGE_AI/coco_validation_jsons/person_keypoints_val2014.json'
### Predictions File:
# predictsFile = '/home/javier/cocoapi/results/instances_val2014_fakebbox100_results.json'
# predictsFile = '/home/javier/cocoapi/results/instances_val2014_fakesegm100_results.json'
# predictsFile = '/home/javier/cocoapi/results/person_keypoints_val2014_fakekeypoints100_results.json'

def main(predictsFile, validFile, annType, iouThrs=(0.5,0.75,0.05)):
    """evaluates from two json files (coco format). We could also do it on he fly, but since
    the evaluation should be done with a bunch of images at the same time, this is probably
    the first and main option to use.

    The Step (third position) should be 0.05. Otherwise the AP might be = -1
    This could probably be solved by changing the calculation of the IoU.
    Nevertheless the mean calculation will only be done with the good values,
    so it shouldn¡t affect much, just that some values will be missed in the
    calculation of the mean.

    It has been tested that this module works for any annotation type of:
    annType = ['segm', 'bbox', 'keypoints']"""
    cocoGt=COCO(validFile)
    cocoDt=cocoGt.loadRes(predictsFile)
    imgIds=sorted(cocoGt.getImgIds())
    print("len imgs Ids", len(imgIds))
    ### added Javi: ...  to avoid:  IndexError: list index out of range on small datasets
    min_imgs = min(len(imgIds), 100)
    imgIds=imgIds[0:min_imgs]
    imgId = imgIds[np.random.randint(min_imgs)]
    # imgIds=imgIds[0:100]
    # imgId = imgIds[np.random.randint(100)]
    ### running evaluation
    cocoEval = cocoeval_modif.COCOeval(cocoGt, cocoDt, annType, iouThrs)
    cocoEval.params.imgIds  = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    stats, stats_dicto = cocoEval.summarize()
    print("\nstats: ", stats)
    ### ... return a list & dicto would be enough, but let's add some sugar:
    df_AP = pd.DataFrame(stats_dicto['AP']).T
    df_AR = pd.DataFrame(stats_dicto['AR']).T
    ### Show data frames in nice format:
    try:
        from tabulate import tabulate
        df_AP['result'] = df_AP['result'].astype(float).round(3)
        # df_AP_T = df_AP.T
        pdtabulate=lambda df_AP:tabulate(df_AP, headers='keys', tablefmt='psql')  # 'psql')  # pipe
        print("\nAP:")
        print(pdtabulate(df_AP))
        df_AR['result'] = df_AR['result'].astype(float).round(3)
        # df_AR_T = df_AR.T
        pdtabulate=lambda df_AR:tabulate(df_AR, headers='keys', tablefmt='psql')  # 'psql')  # pipe
        print("\nAR:")
        print(pdtabulate(df_AR))
    except Exception as e:
        print("error: \n", e)
        print("\n We strongly recommend to pip install tabulate for visualizing pandas DataFrames in your linux terminal")
        print("...")
        print("\nAP DataFrame: \n", df_AP.T)
        print("\nAR DataFrame: \n", df_AR.T)
    print("\n[INFO]: For the moment we use 100 Max. Detects. for bbox and segmentation and 20 Max. Detects. for Keypoints Detection.")
    print("        If you need something else you might have to change the code in 'evaluate.py' or in 'cocoeval_modified.py'\n")
    # return the same results in 4 different formats (change it in the future, when we know which format is best)
    return stats, stats_dicto, df_AP, df_AR

if __name__ == "__main__":
    root_dir = "/media/javier/JaviHD/coco_dataset_2017/person_dog_coco/dataset"
    predictsFile = join(root_dir, "output/coco_instances_results.json")
    validFile = join(root_dir, "annotations/test.json")
    # predictsFile = "/home/ubuntu/dataset/output/coco_instances_results.json"
    #predictsFile = "/home/javier/cocoapi/results/person_keypoints_val2014_fakekeypoints100_results.json"

    # validFile = "/home/ubuntu/dataset/annotations/test_mini.json"
    #validFile = "/home/javier/CLEAR_IMAGE_AI/coco_validation_jsons/person_keypoints_val2014.json"
    annType = 'bbox' # "keypoints"
    #annType = "keypoints"
    iouThrs = (0.5, 0.75, 0.05)
    stats, stats_dicto, df_AP, df_AR = main(predictsFile, validFile, annType, iouThrs)

    # print()
    # logging.info(f"stats: {stats}\n")
    # logging.info(f"stats_dicto: {stats_dicto}\n")
    # logging.info(f"df_AP: {df_AP}\n")
    # logging.info(f"df_AR: {df_AR}\n")


    #########################################################################
    #                           TO DO:                                      #
    #                        ------------                                   #
    #                                                                       #
    # *** Return AP per Class!                                              #
    #  /home/ubuntu/AutoTrainingPipeline/evaluation/cocoapi_ClearImageAI/   #
    #   ...cocoapi/PythonAPI/pycocotools/                                   #
    #            - cocoeval_modif.py                                        #
    #            - evaluate.py                                              #
    # also check ~/anaconda3/envs/pytorch_p36/lib/python3.6/site-packages/  #
    #   ... detectron2/evaluation     in detectron2 machine,                #
    #                         for more info and inspiration                 #
    #########################################################################
