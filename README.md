<br>
<p align="justify"><b>This tutorial aims to cluster frames in cMD trajectories according to protein PCA vectors obtained with GROMACS analysis tools. </b></p>

<p align="justify"> It requires a cMD *xtc trajectory file, a *tpr and a *gro file. In the first part, the trajectory is corrected with gmx trjconv and the PCA vectors are calculated. Then, the pca_dbscan_gmm.py uses DBSCAN to identify outliers and performs a clustering analysis with Gaussian mixture models. The script is also able to identify the PCA vectors that correspond to the highest density of frames in a cluster, through a kernel density-based method, and output the 5 frames that are closest to the identified point. In the second part, the frames are extracted from the trajectory to separate *.gro files. </p>

---


<br>
<h2> <p align="center"> <b>I - Trajectory correction and PCA vector extraction</b> </p></h2>

<br/>

We start by correcting the cMD trajectory using trjconv (this assumes a octahedron box, change the routine according to your box type):

```js
gmx make_ndx -f ref.gro -o act.ndx    # Make a *ndx selection with the region of interest for the analysis. In this case we can use the heavy atoms of all protein residues within 10 angstrom from the catalytic residues.
echo 0 | gmx trjconv -f trajectory.xtc -s ref.tpr -ur compact -pbc mol -center -o trajectory_pbc.xtc    # Correct the PBC of the octahedron box
echo 27 0 | gmx trjconv -f trajectory_pbc.xtc -s ref.gro -fit rot+trans -o trajectory_fit.xtc    # Fit the trajectory relative to the previously created selection.
```

---






