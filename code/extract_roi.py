import os
import argparse
import subprocess
import numpy as np
import pandas as pd
from scipy import io


def add_argparse_args(parent_parser):
    parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)

    parser.add_argument('--subject', type=str)

    return parser


def extract_label(data):
    label_list = []
    vertices_label = []
    for i in range(len(data['data'])):
        temp = data['data'][i][2][0]
        vertices_label.append(temp)
        if temp not in label_list:
            label_list.append(temp)
    return label_list, vertices_label


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser = add_argparse_args(parser)
    args = parser.parse_args()

    num_vertices = {
        'S1': 300245,
        'S2': 270826,
        'S3': 306598,
        'S4': 284718,
        'S5': 280414,
        'S6': 295579,
        'S7': 290278,
        'S8': 258073
    }

    hemi_vertices = {
        'S1': [149079, 151166],
        'S2': [135103, 135723],
        'S3': [155295, 151303],
        'S4': [141922, 142796],
        'S5': [141578, 138836],
        'S6': [146440, 149139],
        'S7': [145747, 144531],
        'S8': [129958, 128115]
    }

    subjects = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
    for subject in subjects:
        fs_root = '/public_bme/data/lishr/Cross_modal/subjects'
        # processed_root = '/public_bme/data/lishr/Cross_modal/Processed_Data'
        processed_root = '/public_bme2/bme-liyuanning/lishr/Cross_modal/Processed_Data'
        sub_analysis_root = os.path.join(processed_root, subject, 'Stimulus', 'basic_analysis')
        sub_roi_root = os.path.join(processed_root, subject, 'ROI')

        if not os.path.exists(sub_roi_root):
            os.makedirs(sub_roi_root)

        os.system('module add apps/afni')
        # Cluster-based correction for ttest results
        ttest_lh = os.path.join(sub_analysis_root, subject+'_match_image_lh.csv')
        ttest_rh = os.path.join(sub_analysis_root, subject+'_match_image_rh.csv')
        if os.path.exists(ttest_lh) and os.path.exists(ttest_rh):
            lh_smoothwm = os.path.join(fs_root, subject[3:], 'SUMA', 'lh.smoothwm.gii')
            rh_smoothwm = os.path.join(fs_root, subject[3:], 'SUMA', 'rh.smoothwm.gii')
            ttest_lh_1D = os.path.join(sub_analysis_root, subject + '_match_image_lh.1D.dset')
            ttest_rh_1D = os.path.join(sub_analysis_root, subject + '_match_image_rh.1D.dset')
            enhance_lh = os.path.join(sub_roi_root, 'enhance_lh_cluster.1D')
            enhance_rh = os.path.join(sub_roi_root, 'enhance_rh_cluster.1D')
            suppression_lh = os.path.join(sub_roi_root, 'suppression_lh_cluster.1D')
            suppression_rh = os.path.join(sub_roi_root, 'suppression_rh_cluster.1D')

            subprocess.run([
                'ConvertDset', '-o_1D', '-input', ttest_lh, '-prefix', ttest_lh[:-3], '-overwrite'
            ])

            subprocess.run([
                'ConvertDset', '-o_1D', '-input', ttest_rh, '-prefix', ttest_rh[:-3], '-overwrite'
            ])

            subprocess.run([
                'SurfClust', '-i', lh_smoothwm, '-input', ttest_lh_1D, '0', '-rmm', '-1.000000',
                '-thresh_col', '3', '-thresh', '0.950000', '-amm2', '20', '-sort_area', '-no_cent',
                '-prefix', enhance_lh, '-out_clusterdset', '-out_roidset', '-out_fulllist', '-overwrite'
            ])

            subprocess.run([
                'SurfClust', '-i', rh_smoothwm, '-input', ttest_rh_1D, '0', '-rmm', '-1.000000',
                '-thresh_col', '3', '-thresh', '0.950000', '-amm2', '20', '-sort_area', '-no_cent',
                '-prefix', enhance_rh, '-out_clusterdset', '-out_roidset', '-out_fulllist', '-overwrite'
            ])

            subprocess.run([
                'SurfClust', '-i', lh_smoothwm, '-input', ttest_lh_1D, '0', '-rmm', '-1.000000',
                '-thresh_col', '4', '-thresh', '0.950000', '-amm2', '20', '-sort_area', '-no_cent',
                '-prefix', suppression_lh, '-out_clusterdset', '-out_roidset', '-out_fulllist', '-overwrite'
            ])

            subprocess.run([
                'SurfClust', '-i', rh_smoothwm, '-input', ttest_rh_1D, '0', '-rmm', '-1.000000',
                '-thresh_col', '4', '-thresh', '0.950000', '-amm2', '20', '-sort_area', '-no_cent',
                '-prefix', suppression_rh, '-out_clusterdset', '-out_roidset', '-out_fulllist', '-overwrite'
            ])

        # Extract ROI - Suppression & Enhance & Insignificant
        ttest_lh = os.path.join(sub_analysis_root, subject+'_match_image_lh.csv')
        ttest_rh = os.path.join(sub_analysis_root, subject+'_match_image_rh.csv')
        if os.path.exists(ttest_lh) and os.path.exists(ttest_rh):
            suppression = []
            enhance = []
            insignificant = []

            count = -1

            with open(os.path.join(sub_roi_root, 'enhance_lh_cluster_ClstMsk_e1_a20.0.1D.dset'), 'r', encoding='gbk') as f:
                enhance_content = f.readlines()
            f.close()

            with open(os.path.join(sub_roi_root, 'suppression_lh_cluster_ClstMsk_e1_a20.0.1D.dset'), 'r', encoding='gbk') as f:
                suppression_content = f.readlines()
            f.close()

            for index in range(len(enhance_content)):
                if '#' in enhance_content[index] or len(enhance_content[index].split()) < 1:
                    continue
                count += 1
                enhance_temp = int((enhance_content[index].split())[0])
                suppression_temp = int((suppression_content[index].split())[0])
                if enhance_temp != 0:
                    enhance.append(count)
                elif suppression_temp != 0:
                    suppression.append(count)
                else:
                    insignificant.append(count)

            with open(os.path.join(sub_roi_root, 'enhance_rh_cluster_ClstMsk_e1_a20.0.1D.dset'), 'r', encoding='gbk') as f:
                enhance_content = f.readlines()
            f.close()

            with open(os.path.join(sub_roi_root, 'suppression_rh_cluster_ClstMsk_e1_a20.0.1D.dset'), 'r', encoding='gbk') as f:
                suppression_content = f.readlines()
            f.close()

            for index in range(len(enhance_content)):
                if '#' in enhance_content[index] or len(enhance_content[index].split()) < 1:
                    continue
                count += 1
                enhance_temp = int((enhance_content[index].split())[0])
                suppression_temp = int((suppression_content[index].split())[0])
                if enhance_temp != 0:
                    enhance.append(count)
                elif suppression_temp != 0:
                    suppression.append(count)
                else:
                    insignificant.append(count)

            suppression = np.array(suppression)
            savedir = os.path.join(sub_roi_root, 'Whole_brain_Suppression.txt')
            np.savetxt(savedir, suppression, fmt='%d')

            enhance = np.array(enhance)
            savedir = os.path.join(sub_roi_root, 'Whole_brain_Enhance.txt')
            np.savetxt(savedir, enhance, fmt='%d')

            insignificant = np.array(insignificant)
            savedir = os.path.join(sub_roi_root, 'Whole_brain_Insignificant.txt')
            np.savetxt(savedir, insignificant, fmt='%d')

        suppression = suppression.tolist()
        enhance = enhance.tolist()
        insignificant = insignificant.tolist()

        print(len(suppression) + len(enhance) + len(insignificant))

        # Extract ROI - EVC
        print('Extract ROI - EVC')
        evc_lh_dir = os.path.join(sub_roi_root, 'Visual_123_lh.1D.roi')
        evc_rh_dir = os.path.join(sub_roi_root, 'Visual_123_rh.1D.roi')
        if os.path.exists(evc_lh_dir) and os.path.exists(evc_rh_dir):
            evc = []

            with open(evc_lh_dir, 'r', encoding='gbk') as f:
                content = f.readlines()
            f.close()

            flag = 1
            for index, line in enumerate(content):
                if 'Label' in line and 'ecc' in line:
                    flag = 0
                if 'Label' in line and 'ecc' not in line:
                    flag = 1
                if '#' in line:
                    continue
                temp = line.split()
                if len(temp) != 2:
                    continue
                if flag:
                    evc.append(int(temp[0]))

            with open(evc_rh_dir, 'r', encoding='gbk') as f:
                content = f.readlines()
            f.close()

            flag = 1
            for index, line in enumerate(content):
                if 'Label' in line and 'ecc' in line:
                    flag = 0
                if 'Label' in line and 'ecc' not in line:
                    flag = 1
                if '#' in line:
                    continue
                temp = line.split()
                if len(temp) != 2:
                    continue
                if flag:
                    evc.append(int(temp[0]) + hemi_vertices[subject][0])

            evc = np.unique(np.array(sorted(evc)))
            savedir = os.path.join(sub_roi_root, 'EVC.txt')
            np.savetxt(savedir, evc, fmt='%d')
        else:
            print('EVC ROI does not exist!')

        print('Extract ROI - V1 V2 V3')
        evc_lh_dir = os.path.join(sub_roi_root, 'Visual_123_lh.1D.roi')
        evc_rh_dir = os.path.join(sub_roi_root, 'Visual_123_rh.1D.roi')
        if os.path.exists(evc_lh_dir) and os.path.exists(evc_rh_dir):
            for label in ['V1', 'V2', 'V3']:
                evc = []

                with open(evc_lh_dir, 'r', encoding='gbk') as f:
                    content = f.readlines()
                f.close()

                flag = 0
                for index, line in enumerate(content):
                    if 'Label' in line and label in line:
                        flag = 1
                    if 'Label' in line and label not in line:
                        flag = 0
                    if '#' in line:
                        continue
                    temp = line.split()
                    if len(temp) != 2:
                        continue
                    if flag:
                        evc.append(int(temp[0]))

                with open(evc_rh_dir, 'r', encoding='gbk') as f:
                    content = f.readlines()
                f.close()

                flag = 0
                for index, line in enumerate(content):
                    if 'Label' in line and label in line:
                        flag = 1
                    if 'Label' in line and label not in line:
                        flag = 0
                    if '#' in line:
                        continue
                    temp = line.split()
                    if len(temp) != 2:
                        continue
                    if flag:
                        evc.append(int(temp[0]) + hemi_vertices[subject][0])

                evc = np.unique(np.array(sorted(evc)))
                savedir = os.path.join(sub_roi_root, label + '.txt')
                np.savetxt(savedir, evc, fmt='%d')
        else:
            print('EVC ROI does not exist!')

        # Extract ROI - EVC Suppression & EVC w/o Suppression
        print('Extract ROI - EVC Suppression & EVC w/o Suppression')
        evc_dir = os.path.join(sub_roi_root, 'EVC.txt')
        ttest_lh = os.path.join(sub_analysis_root, subject+'_match_image_lh.csv')
        ttest_rh = os.path.join(sub_analysis_root, subject+'_match_image_rh.csv')
        if os.path.exists(evc_dir) and os.path.exists(ttest_lh) and os.path.exists(ttest_rh):
            evc_suppression = []
            evc_wo_suppression = []

            lh = pd.read_csv(ttest_lh)
            rh = pd.read_csv(ttest_rh)
            ttest = pd.concat([lh, rh], ignore_index=True)

            t_stat = ttest['t_stat']
            p_value = ttest['p_value']

            evc = np.loadtxt(evc_dir)
            evc = evc[evc != -1].astype(int)

            for i in range(len(evc)):
                if evc[i] in suppression:
                    evc_suppression.append(evc[i])
                else:
                    evc_wo_suppression.append(evc[i])

            evc_suppression = np.array(sorted(evc_suppression))
            savedir = os.path.join(sub_roi_root, 'EVC_Suppression.txt')
            np.savetxt(savedir, evc_suppression, fmt='%d')

            evc_wo_suppression = np.array(sorted(evc_wo_suppression))
            savedir = os.path.join(sub_roi_root, 'EVC_Without_Suppression.txt')
            np.savetxt(savedir, evc_wo_suppression, fmt="%d")

        # Extract ROI - IFS Suppression & Enhance & Insignificant
        print('Extract ROI - IFS Suppression & Enhance & Insignificant')
        data = io.loadmat(os.path.join(processed_root, subject, 'aparc_label.mat'))
        label_list, vertices_label = extract_label(data)
        label_list = sorted(label_list)
        print(len(vertices_label))

        label_dict = {}
        for index, label in enumerate(label_list):
            label_dict[label] = index

        label_count = np.zeros(len(label_list))
        for v in range(len(vertices_label)):
            label_count[label_dict[vertices_label[v]]] += 1

        ifs = []
        for index, label in enumerate(vertices_label):
            if label == 'S_front_inf':
                ifs.append(index)
        ifs = np.array(ifs)
        savedir = os.path.join(sub_roi_root, 'IFS.txt')
        np.savetxt(savedir, ifs, fmt='%d')

        ifs_suppression = []
        ifs_enhance = []
        ifs_insignificant = []
        for i in range(len(ifs)):
            if ifs[i] in suppression:
                ifs_suppression.append(ifs[i])
            elif ifs[i] in enhance:
                ifs_enhance.append(ifs[i])
            else:
                ifs_insignificant.append(ifs[i])

        ifs_suppression = np.array(ifs_suppression)
        savedir = os.path.join(sub_roi_root, 'IFS_Suppression.txt')
        np.savetxt(savedir, ifs_suppression, fmt='%d')

        ifs_enhance = np.array(ifs_enhance)
        savedir = os.path.join(sub_roi_root, 'IFS_Enhance.txt')
        np.savetxt(savedir, ifs_enhance, fmt='%d')

        ifs_insignificant = np.array(ifs_insignificant)
        savedir = os.path.join(sub_roi_root, 'IFS_Insignificant.txt')
        np.savetxt(savedir, ifs_insignificant, fmt='%d')

        # Extract ROI - IPS
        print('Extract ROI - IPS')
        data = io.loadmat(os.path.join(processed_root, subject, 'aparc_label.mat'))
        label_list, vertices_label = extract_label(data)
        label_list = sorted(label_list)
        print(len(vertices_label))

        label_dict = {}
        for index, label in enumerate(label_list):
            label_dict[label] = index

        label_count = np.zeros(len(label_list))
        for v in range(len(vertices_label)):
            label_count[label_dict[vertices_label[v]]] += 1

        ips = []
        for index, label in enumerate(vertices_label):
            # if label == 'G_occipital_sup' or label == 'S_oc_sup_and_transversal' or label == 'S_intrapariet_and_P_trans':
            if label == 'G_occipital_sup' or label == 'S_oc_sup_and_transversal':
                ips.append(index)
        ips = np.array(ips)
        savedir = os.path.join(sub_roi_root, 'IPS.txt')
        np.savetxt(savedir, ips, fmt='%d')

        # Extract ROI - MFG
        print('Extract ROI - MFG')
        data = io.loadmat(os.path.join(processed_root, subject, 'aparc_label.mat'))
        label_list, vertices_label = extract_label(data)
        label_list = sorted(label_list)

        label_dict = {}
        for index, label in enumerate(label_list):
            label_dict[label] = index

        label_count = np.zeros(len(label_list))
        for v in range(len(vertices_label)):
            label_count[label_dict[vertices_label[v]]] += 1

        mfg = []
        for index, label in enumerate(vertices_label):
            if label == 'G_front_middle':
                mfg.append(index)
        mfg = np.array(mfg)
        savedir = os.path.join(sub_roi_root, 'MFG.txt')
        np.savetxt(savedir, mfg, fmt='%d')

        # Extract ROI - IOG
        print('Extract ROI - IOG')
        data = io.loadmat(os.path.join(processed_root, subject, 'aparc_label.mat'))
        label_list, vertices_label = extract_label(data)
        label_list = sorted(label_list)

        label_dict = {}
        for index, label in enumerate(label_list):
            label_dict[label] = index

        label_count = np.zeros(len(label_list))
        for v in range(len(vertices_label)):
            label_count[label_dict[vertices_label[v]]] += 1

        IOG = []
        for index, label in enumerate(vertices_label):
            if label == 'G_and_S_occipital_inf':
                IOG.append(index)
        IOG = np.array(IOG)
        savedir = os.path.join(sub_roi_root, 'IOG.txt')
        np.savetxt(savedir, IOG, fmt='%d')

        # Extract ROI - LHIFGorb
        print('Extract ROI - LHIFGorb')
        data = io.loadmat(os.path.join(processed_root, subject, 'aparc_label.mat'))
        label_list, vertices_label = extract_label(data)
        label_list = sorted(label_list)

        IFGorb = []
        for index, label in enumerate(vertices_label):
            if label == 'G_front_inf-Orbital' and index < hemi_vertices[subject][0]:
                IFGorb.append(index)
        IFGorb = np.array(IFGorb)
        savedir = os.path.join(sub_roi_root, 'LHIFGorb.txt')
        np.savetxt(savedir, IFGorb, fmt='%d')

        # Extract ROI - LHIFG
        print('Extract ROI - LHIFG')
        data = io.loadmat(os.path.join(processed_root, subject, 'aparc_label.mat'))
        label_list, vertices_label = extract_label(data)
        label_list = sorted(label_list)

        IFG = []
        for index, label in enumerate(vertices_label):
            if (label in ['G_front_inf-Opercular', 'G_front_inf-Triangul']) and index < hemi_vertices[subject][0]:
                IFG.append(index)
        IFG = np.array(IFG)
        savedir = os.path.join(sub_roi_root, 'LHIFG.txt')
        np.savetxt(savedir, IFG, fmt='%d')

        # Extract ROI - LHMFG
        print('Extract ROI - LHMFG')
        data = io.loadmat(os.path.join(processed_root, subject, 'aparc_label.mat'))
        label_list, vertices_label = extract_label(data)
        label_list = sorted(label_list)

        MFG = []
        for index, label in enumerate(vertices_label):
            if (label in ['G_front_middle']) and index < hemi_vertices[subject][0]:
                MFG.append(index)
        MFG = np.array(MFG)
        savedir = os.path.join(sub_roi_root, 'LHMFG.txt')
        np.savetxt(savedir, MFG, fmt='%d')

        # Extract ROI - LHPostTemp
        print('Extract ROI - LHTemp')
        data = io.loadmat(os.path.join(processed_root, subject, 'aparc_label.mat'))
        label_list, vertices_label = extract_label(data)
        label_list = sorted(label_list)

        Temp = []
        for index, label in enumerate(vertices_label):
            if (label in ['G_temp_sup-G_T_transv', 'G_temp_sup-Lateral', 'G_temp_sup-Plan_polar', 'G_temp_sup-Plan_tempo', 'G_temporal_inf', 'G_temporal_middle']) and index < hemi_vertices[subject][0]:
                Temp.append(index)
        Temp = np.array(Temp)
        savedir = os.path.join(sub_roi_root, 'LHTemp.txt')
        np.savetxt(savedir, Temp, fmt='%d')

        print('Extract ROI - FFA & PPA')
        fLoc_lh_dir = os.path.join(sub_roi_root, 'fLoc_lh.1D.roi')
        fLoc_rh_dir = os.path.join(sub_roi_root, 'fLoc_rh.1D.roi')
        if os.path.exists(fLoc_lh_dir) and os.path.exists(fLoc_rh_dir):
            for label in ['Face', 'Place']:
                floc = []

                with open(fLoc_lh_dir, 'r', encoding='gbk') as f:
                    content = f.readlines()
                f.close()

                flag = 0
                for index, line in enumerate(content):
                    if 'Label' in line and label in line:
                        flag = 1
                    if 'Label' in line and label not in line:
                        flag = 0
                    if '#' in line:
                        continue
                    temp = line.split()
                    if len(temp) != 2:
                        continue
                    if flag:
                        floc.append(int(temp[0]))

                with open(fLoc_rh_dir, 'r', encoding='gbk') as f:
                    content = f.readlines()
                f.close()

                flag = 0
                for index, line in enumerate(content):
                    if 'Label' in line and label in line:
                        flag = 1
                    if 'Label' in line and label not in line:
                        flag = 0
                    if '#' in line:
                        continue
                    temp = line.split()
                    if len(temp) != 2:
                        continue
                    if flag:
                        floc.append(int(temp[0]) + hemi_vertices[subject][0])

                floc = np.unique(np.array(sorted(floc)))
                savedir = os.path.join(sub_roi_root, label + '.txt')
                np.savetxt(savedir, floc, fmt='%d')
        else:
            print('fLoc ROI does not exist!')