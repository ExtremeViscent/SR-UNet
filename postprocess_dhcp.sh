#!/bin/bash
#
# Script to post-process dHCP scans, in preparation for super-resolution
#
# Frantisek Vasa (frantisek.vasa@kcl.ac.uk)

# path to main folder containing dhcp anatomical data
dhcp_anat=/media/hdd/dhcp/rel3_dhcp_anat_pipeline

# outout paths
dhcp_hires=/media/hdd/dhcp/dhcp_hires
dhcp_lores=/media/hdd/dhcp/dhcp_lores

# if output folders don't exist, create them
# high-res (original dHCP resolution)
if [ ! -d ${dhcp_hires} ]; then mkdir ${dhcp_hires}; fi                     # main folder
if [ ! -d ${dhcp_hires}/labels ]; then mkdir ${dhcp_hires}/labels; fi       # labels
if [ ! -d ${dhcp_hires}/images_t2 ]; then mkdir ${dhcp_hires}/images_t2; fi # t2
if [ ! -d ${dhcp_hires}/images_t1 ]; then mkdir ${dhcp_hires}/images_t1; fi # t1
# low-res (resampled to 1mm isotropic)
if [ ! -d ${dhcp_lores} ]; then mkdir ${dhcp_lores}; fi
if [ ! -d ${dhcp_lores}/labels ]; then mkdir ${dhcp_lores}/labels; fi
if [ ! -d ${dhcp_lores}/images_t2 ]; then mkdir ${dhcp_lores}/images_t2; fi
if [ ! -d ${dhcp_lores}/images_t1 ]; then mkdir ${dhcp_lores}/images_t1; fi

# loop over folders in anatomical directory
for s in ${dhcp_anat}/*/; do
	sub=$(basename $s) # subject name only (path stripped from $s)
	#echo ${subj}

	# loop over sessions
	for ss in ${dhcp_anat}/${sub}/*/; do
		ses=$(basename $ss) # session name only (path stripped from $ss)
		#echo ${ses}

			# current working directory (.../subject/session/anat)
			cwd=${dhcp_anat}/${sub}/${ses}/anat
			#echo ${cwd}

			# check for presence of segmentation, T2w and T1w nii.gz files 
			if [[ -f ${cwd}/${sub}_${ses}_desc-drawem9_dseg.nii.gz && -f ${cwd}/${sub}_${ses}_T2w.nii.gz && -f ${cwd}/${sub}_${ses}_T1w.nii.gz ]]; then
				#echo ${sub}_${ses} >> ${dhcp_anat}/../sub_w_seg_t2_t1.txt # count -> 709/782 subjects

				echo "-------------"
				echo ${sub}_${ses}
				echo "-------------"

				### Original resolution (high-res)

				# copy segmentation labels to output folder
				cp \
				${cwd}/${sub}_${ses}_desc-drawem9_dseg.nii.gz \
				${dhcp_hires}/labels/${sub}_${ses}_desc-drawem9_dseg.nii.gz

				# T2w extraction from skull using mask
				fslmaths \
				${cwd}/${sub}_${ses}_T2w.nii.gz \
				-mas ${cwd}/${sub}_${ses}_desc-brain_mask.nii.gz \
				${dhcp_hires}/images_t2/${sub}_${ses}_T2w_brain.nii.gz

				# T1w extraction from skull using mask
				fslmaths \
				${cwd}/${sub}_${ses}_T1w.nii.gz \
				-mas ${cwd}/${sub}_${ses}_desc-brain_mask.nii.gz \
				${dhcp_hires}/images_t1/${sub}_${ses}_T1w_brain.nii.gz

				#### Downsampled to 1mm isotropic (low-res)

				# downsample labels to 1mm isotropic (w/ nearest-neighbour interpolation)
				flirt \
				-in ${dhcp_hires}/labels/${sub}_${ses}_desc-drawem9_dseg.nii.gz \
				-ref ${dhcp_anat}/../dhcp_1mm_ref.nii.gz \
				-out ${dhcp_lores}/labels/${sub}_${ses}_desc-drawem9_dseg_1mm.nii.gz \
				-interp nearestneighbour \
				-applyxfm

				# downsample T2w to 1mm
				flirt \
				-in ${dhcp_hires}/images_t2/${sub}_${ses}_T2w_brain.nii.gz \
				-ref ${dhcp_anat}/../dhcp_1mm_ref.nii.gz \
				-out ${dhcp_lores}/images_t2/${sub}_${ses}_T2w_brain_1mm.nii.gz \
				-applyxfm

				# downsample T1w to 1mm
				flirt \
				-in ${dhcp_hires}/images_t1/${sub}_${ses}_T1w_brain.nii.gz \
				-ref ${dhcp_anat}/../dhcp_1mm_ref.nii.gz \
				-out ${dhcp_lores}/images_t1/${sub}_${ses}_T1w_brain_1mm.nii.gz \
				-applyxfm

			fi # only subjects with seg, T2w and T1w data

	done # loop over sessions	

done # loop over subjects