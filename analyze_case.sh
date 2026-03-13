python3 plot_scalars.py $1/scalars/scalars_s1.h5
python3 plot_fluxes.py $1/averages/averages_s1.h5
mpirun -n $2 python3 plot_slices.py $1/slices/slices_s*.h5
bash make_movie.sh $1
tar cvf data.tar $1/*.[m,p]??
