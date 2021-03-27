from menpo_functions import *
from deep_heatmaps_model_fusion_net import DeepHeatmapsModel
import os
import pickle

# directory for saving predictions
out_dir = '/Users/arik/Desktop/out/'
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

# directory with conventional landmark detection datasets (for bounding box files)
conv_dir = '/Users/arik/Dropbox/a_mac_thesis/face_heatmap_networks/conventional_landmark_detection_dataset/'

# bounding box type for conventional landmark detection datasets (gt / init)
bb_type='init'

# directory with clm models for tuning step
clm_path='pdm_clm_models/clm_models/g_t_all'

# directory with pdm models for correction step
pdm_path='pdm_clm_models/pdm_models/'

# model path
model_path = '/Users/arik/Dropbox/Thesis_dropbox/models/model_train_wiki/model/deep_heatmaps-60000'

# directory containing test sets
data_dir = '/Users/arik/Dropbox/a_mac_thesis/artistic_faces/artistic_face_dataset/'
test_sets = ['all_AF']  # test sets to evaluate


# data_dir = '/Users/arik/Desktop/Thesis_mac/semi_art_sets/semi_art_sets_wiki_train_2/'
# test_sets = [
#     'challenging_set_aug_geom_texture',
#     'common_set_aug_geom_texture',
#     'test_set_aug_geom_texture',
#     'full_set_aug_geom_texture'
# ]


# load heatmap model
heatmap_model = DeepHeatmapsModel(
    mode='TEST', img_path=conv_dir, test_model_path=model_path, menpo_verbose=False, scale=1)

bb_dir = os.path.join(conv_dir, 'Bounding_Boxes')

# predict landmarks for input test sets
for i,test_data in enumerate(test_sets):

    if i == 0:
        reuse=None
    else:
        reuse=True

    out_temp = os.path.join(out_dir, test_data)
    if not os.path.exists(out_temp):
        os.mkdir(out_temp)

    bb_dictionary = load_bb_dictionary(bb_dir, mode='TEST', test_data=test_data)

    img_list = load_menpo_image_list(img_dir=data_dir, train_crop_dir=data_dir, img_dir_ns=data_dir, mode='TEST',
                                     test_data=test_data, bb_type=bb_type, bb_dictionary=bb_dictionary)

    img_list = img_list[:10]
    print test_data + ':' + str(len(img_list)) + ' images'

    preds = heatmap_model.get_landmark_predictions(img_list=img_list, pdm_models_dir=pdm_path, clm_model_path=clm_path,
                                                   reuse=reuse)

    init_lms = preds['E']
    ppdm_lms = preds['ECp']
    clm_lms = preds['ECpT']
    ect_lms = preds['ECT']
    ecptp_jaw_lms = preds['ECpTp_jaw']
    ecptp_out_lms = preds['ECpTp_out']

    filehandler = open(os.path.join(out_temp,'E_lms'),"wb")
    pickle.dump(init_lms,filehandler)
    filehandler.close()

    filehandler = open(os.path.join(out_temp,'ECp_lms'),"wb")
    pickle.dump(ppdm_lms,filehandler)
    filehandler.close()

    filehandler = open(os.path.join(out_temp,'ECpT_lms'),"wb")
    pickle.dump(clm_lms,filehandler)
    filehandler.close()

    filehandler = open(os.path.join(out_temp,'ECT_lms'),"wb")
    pickle.dump(ect_lms,filehandler)
    filehandler.close()

    filehandler = open(os.path.join(out_temp,'ECpTp_jaw_lms'),"wb")
    pickle.dump(ecptp_jaw_lms,filehandler)
    filehandler.close()

    filehandler = open(os.path.join(out_temp,'ECpTp_out_lms'),"wb")
    pickle.dump(ecptp_out_lms,filehandler)
    filehandler.close()

print("\nDone!\n")