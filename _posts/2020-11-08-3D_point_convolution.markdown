---
layout: post
title:  "Deep Learning on 3D Point Clouds - Part IV"
date:   2020-11-08
description: "An analysis of the first point convolution based method for point cloud processing : PointCNN"
---

<div style="font-size: 0.8em; text-align: justify;" markdown=1>


In <a href="#references">[3]</a>, the authors proposed to apply convolution operations directly on the sparse point clouds. Point convolutions are similar to dense  2D or 3D convolutions : a pointwise multiplication is applied between the nearest neighbors of the point being convolved and a kernel followed by an aggregation (summation) to yield a feature encoding; a sliding operation applies the same  operation with the same kernel to the other points. However, **the points irregularity and unorderness** makes the assignment of the kernel vectors to each point inconsistent resulting in a **difficulty** to **learn local patterns**, <a href="#figure1">Figure 1</a>. To overcome this challenge, the authors propose a special convolution operator called $\mathcal{X}$-Conv. This operator has two main steps. First, transformation matrices are applied to neighborhoods. MLP layers are leveraged to learn producing these matrices. Then, the convolution operation is applied on the transformed neighborhoods with the kernel weights. The pre-convolution transformations are called $\mathcal{X}$-transformations. They allow a reordering and a weighting of each neighbor before the convolution. That way, each neighbor will be assigned to the right element (e.g. vector) in the convolution kernel, and "wrong" neighbors which should not intervene in the convolution can be scaled down.

<center>
<div id="figure1">
  <figure  style="width:100%; margin:0;">
  <img src="{{ '/assets/img/pointcnn_fig1.PNG' | prepend: site.baseurl }}" alt="" style=""> 
  <center  style="font-style: initial;"><b>Figure 1.</b>Convolution inputs from point clouds.<br> In (iii) and (iv), the same point cloud but with different points ordering resulting in : $f_{iii} \neq f_{iv}$</center>
</figure>
</div>
</center>
<br>

In <a href="#figure1">Figure 1</a>, $K$ is a kernel applied to four points in a convolution step. As we can see, from iii to iv, the  points ordering change will yield different matrices representing the same point cloud. The convolution operation between $K$ and those matrices will output different results. Thus the inconsistency in applying a convolution operation to unorder data.

### PointCNN : Model

<center>
<div id="figure2">
  <figure  style="width:100%; margin:0;">
  <img src="{{ '/assets/img/pointcnn_fig2.PNG' | prepend: site.baseurl }}" alt="" style=""> 
  <center  style="font-style: initial;"><b>Figure 2. </b>Hierarchical convolution on regular
grids (upper) and point clouds (lower).
</center>
</figure>
</div>
</center>
<br>
PointCNN is a hierarchical architecture. Each layer is a $\mathcal{X}$-Conv convolution layer that slides (in a convolution style) through its input points to learn local patterns. The hierarchical aspect of this model ensures building a few set of feature vectors with a high receptive field (the original points involved in building those features). These features can further be processed for classification or segmentation.<br>
The model folds as follow:<br>
* **Points Sampling** : Given an input pointcloud, a subset of representative points are sampled using a Farthest Point Sampling algorithm.
* **Neighborhood Grouping** : Then, we gather for each representative point the k nearest neighbors. The neighborhoods built for each point can sometimes overlap, like in an image convolution where the striding size is lower than the kernel size <a href="#references">[10]</a>.
* **$\mathcal{X}$-conv layer** : A $\mathcal{X}$-conv operator is applied to the neighborhoods to output a unique feature vector for each neighborhood. Note that the $\mathcal{X}$-conv operator parameters (kernel weights) are shared between the neighborhoods.
* After the previous steps, we get a subset of **representative points**, each associated with a richer feature vector encoding a local pattern. These steps together make a layer of the hierarchical architecture. The output of each layer goes o the next layer, until we reach a few set of representative points and their corresponding feature vectors, with a high receptive field.<br>

Note that, the number of representative points sampled at each layer has a link with the striding size. In fact, if we sample, all the points from the input, it is equivalent to having a striding size equal to 1. If we sample $k$ points from $N$ points, the striding size will be on average $\frac{k}{N}$
<br>
The core of this model is the $\mathcal{X}$-conv operator. <a href="#algorithm1">Algorithm 1</a> details the way the convolution operation is done thanks to this operator.

<center>
<div id="algorithm1">
  <figure  style="width:100%; margin:0;">
  <img src="{{ '/assets/img/pointcnn_algo.PNG' | prepend: site.baseurl }}" alt="" style=""> 
  <center  style="font-style: initial;"><b></b>The current neighborhood of <i>p</i> can be cast into a $K × Dim$ points matrix <br> $P = (p_1, p_2, ..., p_K)^T$,<br> and the corresponding features matrix $F = (f_1, f_2, ..., f_K)$ of size $K × C1$ , <br>and $K$ denotes the trainable convolution kernels.
</center>
</figure>
</div>
</center>
<br>
This algorithm processes in **two main steps**. First, given a neighborhood, MLP layers are used to learn **$\mathcal{X}$ transformation** matrices. Then, we apply those matrices to neighbors' features, to permute and weight them. Finally, the **convolution** kernel (learnable weights) is applied to the transformed features. 


The model described above is an **encoder**, which encodes **hierarchically** a point cloud into feature vectors. Depending on the task to perform, the features extracted can be directly used for classification; or, a decoder can be added on top of the encoder, to produce encoding vectors for each point.

<center>
<div id="figure3">
  <figure  style="width:100%; margin:0;">
  <img src="{{ '/assets/img/pointcnn_fig3.PNG' | prepend: site.baseurl }}" alt="" style=""> 
  <center  style="font-style: initial;"><b>Figure 3 .</b>
PointCNN architecture
for classification (a and b) and
segmentation (c)</center>
</figure>
</div>
</center>
<br>
In <a href="#figure3">Figure 3.a</a>, a PointCNN encodes a given point cloud into a unique feature vector that can be used for classification. This **pointcloud is hierarchically downsampled into a single representative point**. There are two cases with this configuration and both of them present issues. **Either, the downsampling from one layer to another is smooth**  so that the downsampling ensures a high receptive field for the feature vector at the end : this gives too many layers in the model which becomes too heavy; **or, the downsampling is agressive** and the training is not efficient, since the neighbors  are quickly distanced from one another in the layers, with not enough 'time' to gather more local information before getting aggregated in the subsequent layers : a risk of a low receptive field.<br>

The authors proposed a **solution** that could be **efficient** (reasonable depth) while maintaining a **high receptive field**. For each representative point in the upper layers, instead of taking the immediate $K$ nearest neighbors, they are sampled from the $D$x$K$ nearest neighbors, <a href="#figure3">Figure 3.b</a>. $D$ is called the dillatation rate. With this solution, without increasing the depth, each representative point will have more chances to see more raw points and not get distanced abruptly in the lower layers; thus, increasing the receptive field of the top layers.<br>
Finally, the point cloud can be encoded into a few set of feature vectors instead of a unique vector. Each of them, is used for classification and their predictions are gathered as in a bagging model.
<br>

For a segmentation task, a decoder is added on top of the PointCNN encoder. This decoder is a PointCNN model itself, with higher reprenstative points sampled than the input points, at each layer, <a href="#figure3">Figure 3.c</a>.



### Experiments

PointCNN was trained and evaluated on classification  and segmentation tasks. In <a href="#table1">Table 1</a>, the performances of PointCNN  in the classification on ModelNet40 <a href="#references">[5]</a> and ShapeNet<a href="#references">[7]</a> datasets, are reported.

<center>
<div style=" display: table; width: 60%" id="table1">
<div markdown="1" style="font-size: 0.8em; width: 30%; align-self: center;  vertical-align: middle; display:table-cell;">
|Models/datasets| ModelNet40 | ScanNet |
|PointNet <a href="#references">[1]</a> | 89.2 | - |
|PointNet++ <a href="#references">[9]</a>| 90.7 | 76.1 |
|DGCNN <a href="#references">[9]</a> | 92.2 | - |
|**PointCNN** | 92.2 | 79.7 |

</div>
</div>
<center style="font-style: initial;"><b>Table 1</b> : Classification on ModelNet40 and ShapeNet</center>
</center>
<br>
Using only XYZ coordinates, and with about 1024 points sampled from objects surfaces, PointCNN could outperform the point based benchmark at that time. On the ModelNet40 dataset, another experiment has been conducted consisting in randomly rotating the objects in the horizontal axis to change their facing directions. That way, the model can be insensitive to the objects facing directions which is more consistent with real world applications. The model Overall Accuracy increased to $92.5$.<br>

PointCNN was also tested in object part segmentation  and indoor scene segmentations tasks. The datasets used are : **ShapeNet** for object part segmentation, **S3Dis<a href="#references">[8]</a>** and **ScanNet<a href="#references">[6]</a>** for indoor scene segmentation. For ShapeNet, the metric reported is the **pIOU** : the average of the means IOU for each part category (not object category). In S3Dis, I reported the **mIOU** which is the mean IOU or the average of the means IOU for each category of object in the scene. For ScanNet, the **overall accuracy (OA)** on a per voxel basis is reported in <a href="#table2">Table 2</a>.

<center>
<div style=" display: table; width: 60%" id="table2">
<div markdown="1" style="font-size: 0.8em; width: 30%; align-self: center;  vertical-align: middle; display:table-cell;">
|Models/Dataset| ShapeNet | S3DIS | ScanNet|
 DGCNN <a href="#references">[9]</a> | 85.1 | 56.1 | - |
|PointNet <a href="#references">[1]</a> | 83.7 | 47.6 | 73.9 |
|PointNet++ <a href="#references">[4]</a> | 85.1 | - | 84.5 |
|SGPN <a href="#references">[2]</a> | 85.8 | 50.37 | - |
| **PointCNN** | **86.14** | **65.39** | **85.1**|

</div>
</div>
<center style="font-style: initial;"><b>Table 2</b> : Part/Semenatic segmentation  on ShapeNet, S3DIS, ScanNet</center>
</center>
<br>

#### Further Analysis


* **Ablation studies**<br>
To check the effectiveness  of the $\mathcal{X}$-transformation, the authors tested a version of PointCNN without it <a href="#table3">Table 3</a>. There is no permutation of the neighborhoods in this case before the convolution operation. The removal of the $\mathcal{X}$-transformation matrix induces a decrease of the capacity of the models. To fairly compare those models to the original PointCNN, the authors propose to increase either their depth (w/o $\mathcal{X}$-D) or their width - channels dimensions - (w/o $\mathcal{X}$-W).

<center>
<div style=" display: table; width: 60%" id="table3">
<div markdown="1" style="font-size: 0.8em; width: 30%; align-self: center;  vertical-align: middle; display:table-cell;">
||**PointCNN**| w/o $\mathcal{X}$| w/o $\mathcal{X}$-W | w/o $\mathcal{X}$-D |
|**Core Layers**| $\mathcal{X}$-Conv×4| Conv×4 | Conv×4 | Conv×5 |
|**\#Parameter** | 0.6M |0.54M |0.63M |0.61M |
|**Accuracy (%)** | **92.2** | 90.7 | 90.8 |90.7 |

</div>
</div>
<center style="font-style: initial;"><b>Table 3</b> : $\mathcal{X}$-transformation ablation study</center>
</center>
<br>

Unexpectedly, the models without pre-convolution permutations can learn and reach a decent performance. Though the original PointCNN outperform these models.



* **$\mathcal{X}$-transformation actual role** <br>
In the previous lines, it has been said that the pre-convolution transformations permute the neighborhoods before the convolution operations. This would allow the model, to be insensitive to the points unorderness and irregularity in a convolution setting.
To test this property, the authors plotted a T-SNE visualization of the features of 15 representative points in a given layer for a model trained without a $\mathcal{X}$-transformation, <a href="#figure4">Figure 4.a</a>. Note that given the permutation order of the neighbors at the input of $\mathcal{X}$-Conv, the operator will yield a different feature. For each point, the authors then plotted the features obtained with different permutations of the neighbors. As we can see in <a href="#figure4">Figure 4.a</a>, for the same neighborhood, we have different features regarding the permutation at the input.<br>

<center>
<div id="figure4">
  <figure  style="width:100%; margin:0;">
  <img src="{{ '/assets/img/pointcnn_fig4.PNG' | prepend: site.baseurl }}" alt="" style=""> 
  <center  style="font-style: initial;"><b>Figure 3 .</b>T-SNE visualization of features of 15 representative points<br> in a PointCNN w/o $\mathcal{X}$-transformation (a) and in a PointCNN w/ $\mathcal{X}$-transformation (c). <br>(b) represents the features before $\mathcal{X}$-transformations in the second model.
</center>
</figure>
</div>
</center>
<br>
However, with the $\mathcal{X}$-transformation, the T-SNE visualization in <a href="#figure4">Figure 4.c</a> show that for different permutations of the input, we get very similar features. This seems to confirm the insensitivity of the model to the neighborhoods unorderness in the convolution operator.


That being said, as we have seen in the ablation study, the absence of the $\mathcal{X}$-transformation does not abruptly impact the performance of the model. Further investigations need to be done, to understand the underlying properties learned by the model to be able to 'overcome' in a way the unorderness of the points for the convolution.



### Conclusion

PointCNN is a point based deep learning method that apply convolution operations on raw point clouds. The core idea behind PointCNN is the application of a permutation matrix to the neighborhoods before convolving them. This allows the model to be insensitive to the points unorderness. However, without this tranformation, PointCNN performance (e.g. a classification task) is not drastically decreased. There may be some underlying properties also learned by this point based convolution method to overcome the irregularity and the unorderness of the points.




### References
<br>

<textarea id="bibtex_input" style="display:none;">

@misc{pointcnn,
      title={PointCNN: Convolution On X-Transformed Points}, 
      author={Yangyan Li and Rui Bu and Mingchao Sun and Wei Wu and Xinhan Di and Baoquan Chen},
      year={2018},
      eprint={1801.07791},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      pos={3}
}



@INPROCEEDINGS{8099499,
author={R. Q. {Charles} and H. {Su} and M. {Kaichun} and L. J. {Guibas}},
booktitle={2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, title={PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation},
year={2017},
volume={},
number={},
pages={77-85},
doi={10.1109/CVPR.2017.16},
pos={1}}
}

@inproceedings{qi2017pointnet++,
  title={Pointnet++: Deep hierarchical feature learning on point sets in a metric space},
  author={Qi, Charles Ruizhongtai and Yi, Li and Su, Hao and Guibas, Leonidas J},
  booktitle={Advances in neural information processing systems},
  pages={5099--5108},
  year={2017},
  pos={4}
}

@INPROCEEDINGS{modelnet,
  author={ {Zhirong Wu} and S. {Song} and A. {Khosla} and  {Fisher Yu} and  {Linguang Zhang} and  {Xiaoou Tang} and J. {Xiao}},
  booktitle={2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, 
  title={3D ShapeNets: A deep representation for volumetric shapes}, 
  year={2015},
  volume={},
  number={},
  pages={1912-1920},
  doi={10.1109/CVPR.2015.7298801},
pos={5}}

@misc{scannet,
      title={ScanNet: Richly-annotated 3D Reconstructions of Indoor Scenes}, 
      author={Angela Dai and Angel X. Chang and Manolis Savva and Maciej Halber and Thomas Funkhouser and Matthias Nießner},
      year={2017},
      eprint={1702.04405},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
pos={6}
}

@ARTICLE{9025047,  author={Y. {Xu} and S. {Arai} and F. {Tokuda} and K. {Kosuge}},  journal={IEEE Access},   title={A Convolutional Neural Network for Point Cloud Instance Segmentation in Cluttered Scene Trained by Synthetic Data Without Color},   year={2020},  volume={8},  number={},  pages={70262-70269},  doi={10.1109/ACCESS.2020.2978506}, pos={2}}

@article{shapenet,
author = {Yi, Li and Kim, Vladimir G. and Ceylan, Duygu and Shen, I-Chao and Yan, Mengyan and Su, Hao and Lu, Cewu and Huang, Qixing and Sheffer, Alla and Guibas, Leonidas},
title = {A Scalable Active Framework for Region Annotation in 3D Shape Collections},
year = {2016},
issue_date = {November 2016},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {35},
number = {6},
issn = {0730-0301},
url = {https://doi.org/10.1145/2980179.2980238},
doi = {10.1145/2980179.2980238},
journal = {ACM Trans. Graph.},
month = nov,
articleno = {210},
numpages = {12},
keywords = {shape analysis, active learning},
pos={7}
}

@INPROCEEDINGS{s3dis,  author={I. {Armeni} and O. {Sener} and A. R. {Zamir} and H. {Jiang} and I. {Brilakis} and M. {Fischer} and S. {Savarese}},  booktitle={2016 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},   title={3D Semantic Parsing of Large-Scale Indoor Spaces},   year={2016},  volume={},  number={},  pages={1534-1543},  doi={10.1109/CVPR.2016.170},
pos={8}}


@misc{dgcnn,
      title={Dynamic Graph CNN for Learning on Point Clouds}, 
      author={Yue Wang and Yongbin Sun and Ziwei Liu and Sanjay E. Sarma and Michael M. Bronstein and Justin M. Solomon},
      year={2019},
      eprint={1801.07829},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      pos={9}
}

@inproceedings{NIPS2012_c399862d,
 author = {Krizhevsky, Alex and Sutskever, Ilya and Hinton, Geoffrey E},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {F. Pereira and C. J. C. Burges and L. Bottou and K. Q. Weinberger},
 pages = {1097--1105},
 publisher = {Curran Associates, Inc.},
 title = {ImageNet Classification with Deep Convolutional Neural Networks},
 url = {https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf},
 volume = {25},
 year = {2012},
 pos={10}
}


</textarea>

<div class="bibtex_template" style="">
	<table style="border: none; margin-top: -30px;">
		<td style="vertical-align:top; border:none; width: 50px;"> [<span class="pos"></span>]
		</td>
	<td>
	
  <div class="if author" style="font-weight: bold;">	
	<div >
		<span class="if year">
			<span class="year"></span>, 
		</span>
		<span class="author"></span>
		<span class="if url" style="margin-left: 20px">
		  <a class="url" style="color:black; font-size:10px">(view online)</a>
		</span>
		</div>
	</div>
  <div style="">
    <span class="title"></span>
  </div>
</td>
</table>

</div>
<p id="bibtex_display"></p>


</div>
