---
layout: post
title:  "Deep Learning on 3D Point Clouds"
date:   2020-10-28
description: A walkthrough 3D LIDAR point clouds processing in computer vision with Deep Learning methods for semantic segmentation. An in-depth analysis of some architectures and an introduction to Neural Architecture Search in 3D Computer Vision.
---


<p style="text-align: justify; font-size: 0.9em"><span class="dropcap">T</span>he development of 3D data acquisition technologies like LIDAR or RGB-D cameras, the increasing computing capacity and the recent breakthroughs in Deep Learning has favoured the use of 3D data in a variety of fields for perception : robotics, medical imaging, self-driving cars etc. Those data can be representated as point clouds, RGB-D images or meshes. Despite 2D images, 3D data provide a depth information.
<br>In robotics and self-driving cars, perception is crucial. For example, a robot needs to localize the wrist of a glass to be able to grab it; an autonomous vehicle needs to detectthe other vehicles, the traffic lights, the road, to be able to drive. This necessity has given rise to a number of tasks in 3D data processing like object detection, semantic segmentation, instance segmentation for scene understanding, object part segmentation, object  <b>[1]</b>
</p>

### Introduction
<p style="text-align: justify; font-size: 0.9em">A point cloud is a set of un-ordered points lying in a three-dimensional space. Each point in a point cloud can be associated with a feature like a RGB color, a normal vector an intensity etc. Point clouds are generally acquired with sensors like LIDARs. 
Throughout the years, numerous Deep Learning methods have emerged for point clouds processing and can be grouped into the following families:</p>


<p style="text-align: justify; font-size: 0.9em"><b>Multi-View based methods</b><br>In such methods <b>[2]</b>, <b>[3]</b>, <b>[4]</b>, the 3D point clouds are projected onto 2D images from different angles. A 2D convolutional neural network (CNNs) <b>[5]</b> is then leveraged to extract some features from those images which are further processedfor tasks like  classification and even 3D segmentation.</p>



<p style="text-align: justify; font-size: 0.9em"> <b>3D regular Convolution (voxel-based) methods</b><br>They are drawn from the same motivations behind 2D CNNs : learning local patterns. In those methods, 3-dimensional kernels are used to apply regular convolutions directly in the 3D space. However, given the irregularity of the points in a point cloud such 3D Convolutional Neural Networks cannot be directly applied to  the raw points. The 3D space is first voxelized into a regular grid, where each cell is coded in a binary way (empty or not) <b>[]</b>; or is represented by an aggregated vector of the points falling in the same cell or the null vector if empty <b>[]</b>, <b>[]</b>, <b>[]</b>.</p>


<p style="text-align: justify; font-size: 0.9em"> <b>Graph-based methods</b><br>
In this category, the point clouds are represented with a graph structure. <b>[]</b> proposes a method called edge-convolution where a multilayer perceptron MLP is applied to the neighbors of each node and aggregated to build a feature vector for each node. A global max pooling can be applied to the feature vectors to extract a global encoding of the entire point cloud for classification.</p>


<p style="text-align: justify; font-size: 0.9em"> <b>Point based methods</b><br>
Point based methods take as input the raw point clouds as un-ordered sets of points. <b>[]</b>, <b>[]</b> leverage MLPs and max pooling to build a symetric function able to process the raw point clouds and extract some features per point or a global feature encoding the entire point cloud. On the other hand, <b>[]</b>, <b>[]</b> directly apply convolution operations on the points without any voxelization of the space, to learn local patterns. </p>


The list above is not exhaustive. There are many other families of Deep Learning methods like <b>[]</b> where we consider voxel representation + raw point cloud representation as input. Multi-modal approaches consider multiple data sources like LIDAR point clouds + RGB-D images as in <b>[]</b>. <b>[]</b>, <b>[]</b> propose sparse convolutional networks, where the space is voxelized into a sparse 3D grid. The 3D voxel convolutions are applied only on the non-empty cells which is some cases can represent less than 10% of the voxel grids <b>[]</b>.

In this article, let us focus on some Deep learning architectures, their contributions and a deep analysis of their results on the benchmark datasets.


# Heading 1
 \begin{equation} \label{label} a=1 \end{equation} 
 $a=1$ si je le dis.
## Heading 2

### Heading 3

#### Heading 4

##### Heading 5

###### Heading 6

<blockquote>Aenean lacinia bibendum nulla sed consectetur. Morbi leo risus, porta ac consectetur ac, vestibulum at eros. Cras mattis consectetur purus sit amet fermentum. Nulla vitae elit libero, a pharetra augue. Curabitur blandit tempus porttitor. Donec sed odio dui. Cras mattis consectetur purus sit amet fermentum.</blockquote>

Nullam quis risus eget urna mollis ornare vel eu leo. Cras mattis consectetur purus sit amet fermentum. Duis mollis, est non commodo luctus, nisi erat porttitor ligula, eget lacinia odio sem nec elit. Vivamus sagittis lacus vel augue laoreet rutrum faucibus dolor auctor.

## Unordered List
* List Item
* Longer List Item
  * Nested List Item
  * Nested Item
* List Item

## Ordered List
1. List Item
2. Longer List Item
    1. Nested OL Item
    2. Another Nested Item
3. List Item

## Definition List
<dl>
  <dt>Coffee</dt>
  <dd>Black hot drink</dd>
  <dt>Milk</dt>
  <dd>White cold drink</dd>
</dl>

Donec id elit non mi porta gravida at eget metus. Cum sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Maecenas faucibus mollis interdum. Donec sed odio dui. Cras justo odio, dapibus ac facilisis in, egestas eget quam.

Cras justo odio, dapibus ac facilisis in, egestas eget quam. Curabitur blandit tempus porttitor. Cum sociis natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Donec id elit non mi porta gravida at eget metus. Aenean eu leo quam. Pellentesque ornare sem lacinia quam venenatis vestibulum. Sed posuere consectetur est at lobortis. Vivamus sagittis lacus vel augue laoreet rutrum faucibus dolor auctor.

Maecenas faucibus mollis interdum. Maecenas faucibus mollis interdum. Duis mollis, est non commodo luctus, nisi erat porttitor ligula, eget lacinia odio sem nec elit. Etiam porta sem malesuada magna mollis euismod. Vestibulum id ligula porta felis euismod semper. Cras mattis consectetur purus sit amet fermentum.

Sed posuere consectetur est at lobortis. Fusce dapibus, tellus ac cursus commodo, tortor mauris condimentum nibh, ut fermentum massa justo sit amet risus. Fusce dapibus, tellus ac cursus commodo, tortor mauris condimentum nibh, ut fermentum massa justo sit amet risus. Aenean eu leo quam. Pellentesque ornare sem lacinia quam venenatis vestibulum.

Curabitur blandit tempus porttitor. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Vivamus sagittis lacus vel augue laoreet rutrum faucibus dolor auctor. Curabitur blandit tempus porttitor. Nullam quis risus eget urna mollis ornare vel eu leo. Maecenas faucibus mollis interdum. Nullam id dolor id nibh ultricies vehicula ut id elit.


<figure>
	<img src="{{ '/assets/img/touring.jpg' | prepend: site.baseurl }}" alt=""> 
	<figcaption>Fig1. - This is an example figcaption</figcaption>
</figure>

{% highlight html %}
<figure>
	<img src="{{ '/assets/img/touring.jpg' | prepend: site.baseurl }}" alt=""> 
	<figcaption>Fig1. - This is an example figcaption</figcaption>
</figure>
{% endhighlight %}




<p class="intro"><span class="dropcap">Y</span>ou'll find this post in your `_posts` directory - edit this post and re-build (or run with the `-w` switch) to see your changes! To add new posts, simply add a file in the `_posts` directory that follows the convention: YYYY-MM-DD-name-of-post.ext.</p>

Jekyll also offers powerful support for code snippets:

{% highlight ruby %}
def print_hi(name)
  puts "Hi, #{name}"
end
print_hi('Tom')
#=> prints 'Hi, Tom' to STDOUT.
{% endhighlight %}

Check out the [Jekyll docs][jekyll] for more info on how to get the most out of Jekyll. File all bugs/feature requests at [Jekyll's GitHub repo][jekyll-gh].

[jekyll-gh]: https://github.com/mojombo/jekyll
[jekyll]:    http://jekyllrb.com



<h2>References</h2>
<textarea id="bibtex_input" style="display:none;">
@misc{guo2020deep,
      title={Deep Learning for 3D Point Clouds: A Survey}, 
      author={Yulan Guo and Hanyun Wang and Qingyong Hu and Hao Liu and Li Liu and Mohammed Bennamoun},
      year={2020},
      eprint={1912.12033},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      pos={1}
}
}

@inproceedings{su2015multi,
  title={Multi-view convolutional neural networks for 3d shape recognition},
  author={Su, Hang and Maji, Subhransu and Kalogerakis, Evangelos and Learned-Miller, Erik},
  booktitle={Proceedings of the IEEE international conference on computer vision},
  pages={945--953},
  year={2015},
  pos={2}
}
@article{leng20153d,
  title={3D object retrieval with stacked local convolutional autoencoder},
  author={Leng, Biao and Guo, Shuang and Zhang, Xiangyang and Xiong, Zhang},
  journal={Signal Processing},
  volume={112},
  pages={119--128},
  year={2015},
  publisher={Elsevier},
  pos={3}
}

@inproceedings{bai2016gift,
  title={Gift: A real-time and scalable 3d shape search engine},
  author={Bai, Song and Bai, Xiang and Zhou, Zhichao and Zhang, Zhaoxiang and Jan Latecki, Longin},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={5023--5032},
  year={2016},
  pos={4}
}


@INPROCEEDINGS{8099499,
author={R. Q. {Charles} and H. {Su} and M. {Kaichun} and L. J. {Guibas}},
booktitle={2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)}, title={PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation},
year={2017},
volume={},
number={},
pages={77-85},
doi={10.1109/CVPR.2017.16},
pos={-}}
}

</textarea>

<div class="bibtex_template">
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
<div id="bibtex_display"></div>
