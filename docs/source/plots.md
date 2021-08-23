Steps to produce all plots
==========================

Below are the steps to produce all plots in [paper tbd].

## Figure 1: Full Euclid-like and Euclid DR1-like masks

a) Plot the two masks with `plotting.plot_masks`:

```python
[python]

full_euclid_mask_path = 'path-to-Full-Euclid-like-mask.fits.gz'
dr1_mask_path = 'path-to-Euclid-DR1-like-mask.fits.gz'
full_euclid_coord = 'E' # if full Euclid-like mask stored in elliptic coordinates
dr1_coord = 'G' # if Euclid DR1-like mask stored in galactic coordinates

plotting.plot_masks(full_euclid_mask_path, dr1_mask_path, full_euclid_coord=full_euclid_coord, dr1_coord=dr1_coord)
```

## Figure 2: Correlation matrices

a) Produce theory Cls for a single cosmology – as [gaussian_cl_likelihood Fig. 7 step (a)](https://gaussian-cl-likelihood.readthedocs.io/en/latest/source/plots.html#figure-7-illustration-of-l-l-eff-mapping).

b) Create N(z) distribution for input to CosmoLike, using `cosmolike_utils.create_nz`:

```python
[python]

save_path = 'path-to-save-nz.nz'
z_min = 0
z_max = 5.3423
z_steps = 10000
z_mean = [0.65, 0.95, 1.25, 1.55, 1.85]
z_std = 0.3

cosmolike_utils.create_nz(save_path, z_min=z_min, z_max=z_max, z_steps=z_steps, z_mean=z_mean, z_std=z_std)
```

c) Calculate full-sky connected non-Gaussian covariance matrix for the first redshift bin, using [CosmoCov_ClCov](https://github.com/robinupham/CosmoCov_ClCov), or [download the public one](https://doi.org/10.5281/zenodo.5163132).

If calculating from scratch (note this takes a very long time):

1. Copy the [example config file here](https://github.com/robinupham/CosmoCov_ClCov/blob/master/example_input/example.ini) and make the following changes:

    `area : 41235.0` (full sky)

    `# c_footprint_file : ...` (i.e. comment out or remove this line)

    `lmax : 8000`

    `lmin : 2`

    `do_g : 0`

    `do_ss : 0`

    `do_cng : 1`

2. Run `CosmoCov_ClCov`:

    ```shell
    [shell]

    ./get_shear_clcov path-to-config.ini 0
    ```

3. Convert the result to .npz format, using `cosmolike_utils.txt_to_npz':

    ```python
    [python]

    input_filemask = 'path-to-cov-cng.txt'
    fsky = 1

    cosmolike_utils.txt_to_npz(input_filemask, fsky=fsky)
    ```

4. Optionally delete .txt version, using `cosmolike_utils.delete_txt':

    ```python
    [python]

    input_filemask = 'path-to-cov-cng.txt'

    cosmolike_utils.delete_txt(input_filemask)
    ```

d) Per mask:

1. Calculate mixing matrix, using `post_processing.get_mixmat`:

    ```python
    [python]

    mask_path = 'path-to-mask.fits.gz'
    nside = 4096
    lmax_mix = 8000
    lmax_out = 5000
    save_path = 'path-to-save-mixmat.npz'

    post_processing.get_mixmat(mask_path, nside, lmax_mix, lmax_out, save_path)
    ```

2. Calculate Gaussian covariance matrix for the first redshift bin, using `gaussian_cov.get_cov_blocks`:

    ```python
    [python]

    n_zbin = 1
    she_she_input_filemask = 'path-to-theory-cls/shear_cl/bin_{hi_zbin}_{lo_zbin}.txt'
    lmax_in = 8000
    lmin_in = 2
    she_nl_path = None
    noise_lmin = 0
    mask_path = 'path-to-mask.fits.gz'
    nside = 4096
    lmax_mix = lmax_in
    lmax_out = 5000
    lmin_out = 2
    save_path = 'path-to-save-block-{spec1_idx}_{spec2_idx}.npz'

    gaussian_cov.get_cov_blocks(n_zbin, she_she_input_filemask, lmax_in, lmin_in, she_nl_path, noise_lmin, mask_path, nside, lmax_mix, lmax_out, lmin_out, save_path)
    ```

3. Calculate mask power spectrum for input to CosmoLike, using `cosmolike_utils.get_mask_cl`:

    ```python
    [python]

    mask_path = 'path-to-mask.fits.gz'
    save_path = 'path-to-save-mask-cl.dat'
    lmax = 1000 # anything more is ignored by CosmoLike
    nside = 4096

    cosmolike_utils.get_mask_cl(mask_path, save_path, lmax, nside)
    ```

4. Calculate super-sample covariance matrix for the first redshift bin using [CosmoCov_ClCov](https://github.com/robinupham/CosmoCov_ClCov), by repeating steps (c) 1--4 with the following changes:

    `area : {actual mask area in square degrees}` (np.mean(mask) * 41235.0)

    `c_footprint_file : path-to-mask-cl.dat`

    `do_ss : 1`

    `do_cng : 0`

5. Measure tomographic Cls for this mask from the [Takahashi et al. HSC simulations](http://cosmo.phys.hirosaki-u.ac.jp/takahasi/allsky_raytracing/), using `simulation.loop_realisations`:

    ```python
    [python]

    first_real = 0
    last_real = 107 # or less, e.g. to split over multiple machines
    z_mean = [0.65, 0.95, 1.25, 1.55, 1.85]
    z_std = 0.3
    mask_path = 'path-to-mask.fits.gz'
    slice_path = 'tmp-path-to-slice_allskymap_nres12{realisation_id}.{slice_id}.mag.dat'
    lmax = 5000
    lmin = 2
    cl_save_path = 'path-to-save-cls_{realisation_id}.npz'
    delete_after_use = True # slice files are 3 GB, and there are 38 x 108 of them

    simulation.loop_realisations(first_real, last_real, z_mean, z_std, mask_path, slice_path, lmax, lmin, cl_save_path, delete_after_use=delete_after_use)
    ```

6. Combine simulated Cls into a single file, using `simulation.combine_sim_cl`:

    ```python
    [python]

    input_filemask = 'path-to-save-cls_{realisation}.npz'
    n_real = 108
    n_zbin = 5
    lmax = 5000
    lmin = 2
    output_path = 'path-to-save-all-cls.npz'

    simulation.combine_sim_cl(input_filemask, n_real, n_zbin, lmax, lmin, output_path)
    ```

e) Compute all correlation matrices, using `plot_utils.get_cov_mats`:

```python
[python]

cov_cng_fullsky_path = 'path-to-full-sky-cng-cov-from-step-b.npz'
theory_cl_path = 'path-to-theory-cls/shear_cl/bin_1_1.txt'
per_mask_data = [
    {
        'mask_label': 'Full sky',
        'fsky': 1.0,
        'mixmat_path': None,
        'sim_cl_path': 'path-to-all-sim_cl_fullsky.npz',
        'cov_g_path': 'path-to-cov_g_fullsky.npz',
        'cov_ss_path': 'path-to-cov_ss_fullsky.npz'
    },
    {
        'mask_label': 'Full Euclid-like',
        'fsky': 0.302,
        'mixmat_path': 'path-to-mixmat_fulleuclid.npz',
        'sim_cl_path': 'path-to-all-sim_cl_fulleuclid.npz',
        'cov_g_path': 'path-to-cov_g_fulleuclid.npz',
        'cov_ss_path': 'path-to-cov_ss_fulleuclid.npz'
    },
    {
        'mask_label': 'Euclid DR1-like',
        'fsky': 0.062,
        'mixmat_path': 'path-to-mixmat_dr1.npz',
        'sim_cl_path': 'path-to-all-sim_cl_dr1.npz',
        'cov_g_path': 'path-to-cov_g_dr1.npz',
        'cov_ss_path': 'path-to-cov_ss_dr1.npz'
    }
]
lmin = 2
lmax = 5000
lmax_mix = 8000
save_path = 'path-to-save-output.npz'

plot_utils.get_cov_mats(cov_cng_fullsky_path, theory_cl_path, per_mask_data, lmin, lmax, lmax_mix, save_path)
```

f) Plot correlation matrices, using `plotting.cov_mats`:
```python
[python]

data_path = 'path-to-output-from-step-e.npz'
lmax_plot = 3000
lmin = 2
row_order = ['corr_g', 'corr_ss', 'corr_cng', 'corr_tot', 'sim_corr']
row_labels = ['G', 'SS', 'CNG', 'Total', 'Simulations']
downsample_fac = 30

plotting.cov_mats(data_path, lmax_plot, lmin, row_order, row_labels, downsample_fac=downsample_fac)
```

## Figure 3: Covariance diagonals

a) As Fig. 2 steps (a)-(d).

b) Compute correlation matrices and extract diagonals, using `plot_utils.get_cov_diags`:
```python
[python]

diags = [0, 2, 10, 100]
# All other parameters identical to Fig. 2 step (e)

plot_utils.get_cov_diags(cov_cng_fullsky_path, theory_cl_path, per_mask_data, lmin, lmax, lmax_mix, diags, save_path)
```

c) Plot diagonals, using `plotting.cov_diags`:
```python
[python]

data_path = 'path-to-output-from-step-b'
lmax = 5000
lmin = 2
lmax_plot = 3000
roll_window_size = 50

plotting.cov_diags(data_path, lmax, lmin, lmax_plot, roll_window_size)
```

## Figure 4: Gaussian covariance diagonals

a) Produce theory Cls for a single cosmology - as Fig. 2 step (a)

b) Per mask:

1. Calculate Gaussian covariance (lower resolution than Figs. 2-3), using `gaussian_cov.get_cov_blocks`:

    ```python
    [python]

    n_zbin = 1
    she_she_input_filemask = 'path-to-theory-cls/shear_cl/bin_{hi_zbin}_{lo_zbin}.txt'
    nside = 1024
    lmax_in = 3 * nside - 1
    lmin_in = 2
    she_nl_path = None
    noise_lmin = 0
    mask_path = 'path-to-mask.fits.gz'
    lmax_mix = lmax_in
    lmax_out = 1000
    lmin_out = 2
    save_path = 'path-to-save-block-{spec1_idx}_{spec2_idx}.npz'

    gaussian_cov.get_cov_blocks(n_zbin, she_she_input_filemask, lmax_in, lmin_in, she_nl_path, noise_lmin, mask_path, nside, lmax_mix, lmax_out, lmin_out, save_path)
    ```

2. Produce simulated Cls from a single Gaussian field, using `simulation.gaussian_sim`:

    ```python
    [python]

    cl_in_path = 'path-to-theory-cls/shear_cl/bin_1_1.txt'
    nside = 1024
    lmax_in = 3 * nside - 1
    lmin_in = 2
    mask_path = 'path-to-mask.fits.gz'
    lmax_out = 1000
    lmin_out = 2
    n_real = 108
    save_path = 'path-to-save-sim-cl.npz'

    simulation.gaussian_sim(cl_in_path, lmax_in, lmin_in, nside, mask_path, lmax_out, lmin_out, n_real, save_path)
    ```

c) Compute correlation matrices and extract diagonals, using `plot_utils.get_cov_diags_gaussian`:

```python
[python]

per_mask_data = [
    {
        'mask_label': 'Full sky',
        'sim_cl_path': 'path-to-gaussian-sim-cl_fullsky.npz',
        'cov_g_path': 'path-to-cov_g-lmax1000_fullsky.npz'
    },
    {
        'mask_label': 'Full Euclid-like',
        'sim_cl_path': 'path-to-gaussian-sim-cl_fulleuclid.npz',
        'cov_g_path': 'path-to-cov_g-lmax1000_fulleuclid.npz'
    },
    {
        'mask_label': 'Euclid DR1-like',
        'sim_cl_path': 'path-to-gaussian-sim-cl_dr1.npz',
        'cov_g_path': 'path-to-cov_g-lmax1000_dr1.npz'
    }
]
diags = [0, 2, 10, 100]
save_path = 'path-to-save-output.npz'

plot_utils.get_cov_diags_gaussian(per_mask_data, diags, save_path)
```

d) Plot diagonals, using `plotting.cov_gaussian`:
```python
[python]

data_path = 'path-to-output-from-step-c'
lmax = 1000
lmin = 2
roll_window_size = 50

plotting.cov_gaussian(data_path, lmax, lmin, roll_window_size)
```

## Figure 5: Covariance diagonals with noise

a) Obtain shear noise power spectrum - as [gaussian_cl_likelihood Fig. 1 step (b)](https://gaussian-cl-likelihood.readthedocs.io/en/latest/source/plots.html#figure-1-histograms-of-wishart-and-gaussian-1d-posterior-maxima-and-per-realisation-difference).

b) As Fig. 2 steps (a)--(d), with two differences:

1. Don't do the simulation steps (step (c) parts 5--6).

2. Include the noise power spectrum in the call to `gaussian_cov.get_cov_blocks`:

    ```python
    she_nl_path = 'path-to-shear-noise-cls.txt'
    noise_lmin = 2
    ```
c) Compute correlation matrices and extract diagonals, using `plot_utils.get_cov_diags_withnoise`:

```python
[python]

# All parameters identical to Fig. 3 step (b),
# except to refer to the Gaussian covariances with noise from step (b) part 2 above.

plot_utils.get_cov_diags_withnoise(cov_cng_fullsky_path, per_mask_data, lmin, lmax, lmax_mix, diags, save_path)
```

d) Plot diagonals, using `plotting.cov_withnoise`:
```python
[python]

data_path = 'path-to-output-from-step-c'
lmax_plot = 3000

plotting.cov_withnoise(data_path, lmax_plot)
```

## Figure 6: Connected non-Gaussian approximation validation

a) Calculate N(z) distribution - as Fig. 2 step (b).

b) Calculate power spectrum of full Euclid-like mask - as Fig. 2 step (d) part 3.

c) Calculate all blocks of the super-sample covariance matrix for the full Euclid-like mask using [CosmoCov_ClCov](https://github.com/robinupham/CosmoCov_ClCov) - as Fig. 2 step (d) part 4, but repeated for all values of `spec1_idx` from 0 to 14:

```bash
[bash]

for spec1_idx in {0..14} # or split this over multiple machines, but Cov_SS is reasonably fast
do
    ./get_shear_clcov path-to-config.ini spec1_idx
done
```

d) Convert blocks to .npz and optionally delete .txt versions - as Fig. 2 step (c) parts 3-4, but using an appropriate wildcard:

```python
input_filemask = 'cov_ss_spec1_[0-9]*_spec2_[0-9]*.txt'
```

e) Calculate mixing matrix for the full Euclid-like mask - as Fig. 2 step (d) part 1.

f) Apply mixing matrix to all blocks, using `post_processing.mix_blocks`:

```python
[python]

input_dir = 'path-to-directory-containing-cov-ss-blocks'
input_filemask = 'cov_ss_spec1_[0-9]*_spec2_[0-9]*.npz'
mixmat_path = 'path-to-mixing-matrix-from-step-e.npz'
output_dir = 'path-to-output-directory' # must already exist
lmin = 2
lmax_in = 8000
lmax_out = 5000

post_processing.mix_blocks(input_dir, input_filemask, mixmat_path, output_dir, lmin, lmax_in, lmax_out)
```

g) Calculate bandpower binning matrix using [gaussian_cl_likelihood.python.simulation.get_binning_matrix](https://gaussian-cl-likelihood.readthedocs.io/en/latest/source/simulation.html#simulation.get_binning_matrix):

```python
[python]

n_bandpowers = 12
output_lmin = 2
output_lmax = 5000
save_path = 'path-to-save-binning-matrix.npz'

pbl = gaussian_cl_likelihood.python.simulation.get_binning_matrix(n_bandpowers, output_lmin, output_lmax)

header = f'Bandpower binning matrix for {n_bandpowers} log-spaced bandpowers from {output_lmin} to {output_lmax}'
np.savez_compressed(save_path, pbl=pbl, header=header)
```

h) Apply binning matrix to all blocks, using `post_processing.bin_blocks`:

```python
[python]

input_dir = 'path-to-output-directory-from-step-f'
input_filemask = 'cov_ss_spec1_[0-9]*_spec2_[0-9]*.npz'
input_label = 'cov_mixed'
binmat_path = 'path-to-binning-matrix-from-step-g.npz'
output_dir = 'path-to-output-directory'

post_processing.bin_blocks(input_dir, input_filemask, input_label, binmat_path, output_dir)
```

i) Calculate weighted average l per bandpower, using `cng_approx.get_avg_l`:

```python
[python]

lmin = 2
lmax = 5000
n_bandpower = 12

cng_approx.get_avg_l(lmin, lmax, n_bandpower)
```

j) Obtain ratios of approximate to exact covariance for the binning approximation, using `cng_approx.test_bin_weights`:

```python
[python]

ss_block_filemask = 'path-to-unmixed-unbinned-cov_ss-from-step-d/cov_ss_spec1_{spec1_idx}_spec2_{spec2_idx}.npz'
binmat_path = 'path-to-binning-matrix-from-step-g.npz'
lmax = 5000
lmin = 2
ells = [3, 6, 12, 22, 42, 81, 155, 298, 572, 1098, 2108, 4046] # from step i
n_spec = 15
save_path = 'path-to-save-output.npz'

cng_approx.test_bin_weights(ss_block_filemask, binmat_path, lmax, lmin, ells, n_spec, save_path)
```

k) Obtain ratios of approximate to exact covariance for the mixing approximation, using `cng_approx.test_mix_weights`:

```python
[python]

unmixed_unbinned_ss_dir = 'path-to-unmixed-unbinned-cov_ss-from-step-d'
mixed_binned_ss_dir 'path-to-mixed-binned-cov_ss-from-step-h'
input_filemask = 'cov_ss_spec1_{spec1_idx}_spec2_{spec2_idx}.npz'
binmat_path = 'path-to-binning-matrix-from-step-g.npz'
n_spec = 15
save_path = 'path-to-save-output.npz'

cng_approx.test_mix_weights(unmixed_unbinned_ss_dir, mixed_binned_ss_dir, input_filemask, binmat_path, n_spec, save_path)
```

l) Plot histograms of covariance ratios for both approximations, using `plotting.cng_approx`:

```python
[python]

bin_weight_ratios_path = 'path-to-output-from-step-j.npz'
mix_weight_ratios_path = 'path-to-output-from-step-l.npz'

plotting.cng_approx(bin_weight_ratios_path, mix_weight_ratios_path)
```

## Figure 7: 2D posteriors

a) Create two-dimensional `w`-`wa` CosmoSIS grid - as [gaussian_cl_likelihood Fig. 4 step (a)](https://gaussian-cl-likelihood.readthedocs.io/en/latest/source/plots.html#figure-4-2d-posterior-with-discrepant-fiducial-parameters).

b) Create similar CosmoSIS grid of `omega_m`-`sigma_8` - as step (a), but with the following differences:

1. Use the following in the input to [gaussian_cl_likelihood.python.cosmosis_utils.generate_chain_input](https://gaussian-cl-likelihood.readthedocs.io/en/latest/source/cosmosis_utils.html#cosmosis_utils.generate_chain_input):

    ```python
    params = {
        'cosmological_parameters-omega_m': {
            'min': 0.3080,
            'max': 0.3213,
            'steps': 61
        },
        'cosmological_parameters--sigma8_input': {
            'min': 0.8170,
            'max': 0.8347,
            'steps': 61
        }
    }
    ```

2. Add the `sigma8_rescale` module to the CosmoSIS pipeline ini file:

    ```ini
    modules = consistent_parameters camb sigma8_rescale halofit no_bias gaussian_window project_2d

    [sigma8_rescale]
    file = cosmosis-standard-library/utility/sample_sigma8/sigma8_rescale.py
    ```

3. In the CosmoSIS values ini file, set `w` and `wa` to fixed values and set broad ranges for `omega_m` and `sigma8_input`:

    ```ini
    w = -1.0
    wa = 0.0
    omega_m = 0.1 0.3 0.9
    sigma8_input = 0.1 0.8 1.5
    ```

c) Obtain shear noise power spectrum - as [gaussian_cl_likelihood Fig. 1 step (b)](https://gaussian-cl-likelihood.readthedocs.io/en/latest/source/plots.html#figure-1-histograms-of-wishart-and-gaussian-1d-posterior-maxima-and-per-realisation-difference).

d) Calculate bandpower binning matrix - as Fig. 6 step (g).

e) Calculate N(z) distribution - as Fig. 2 step (b).

f) Obtain full-sky connected non-Gaussian covariance matrix for the first redshift bin - as Fig. 2 step (c).

g) Calculate weighted average l per bandpower - as Fig. 6 step (i).

h) Calculate full-sky 5-z-bin connected non-Gaussian covariance for all pairs of weighted average ells obtained in step (f) - as Fig. 2 step (c) with the following changes in the config file:

```ini
# lmax : ...
# lmin : ...
ell : 3,6,12,22,42,81,155,298,572,1098,2108,4046
```
and run for all blocks as Fig. 6 step (c).

i) Calculate binning weights (suitable for any mask), using `cng_approx.get_bin_weights`:

```python
[python]

full_cov_path = 'path-to-full-cng-block-from-step-f.npz'
binmat_path = 'path-to-binning-matrix-from-step-d.npz'
lmax = 5000
lmin = 2
ells = [3, 6, 12, 22, 42, 81, 155, 298, 572, 1098, 2108, 4046]
save_path = 'path-to-save-binning-weights.npz'

cng_approx.get_bin_weights(full_cov_path, binmat_path, lmax, lmin, ells, save_path)
```

j) For each mask:

1. Generate mixing matrix - as Fig. 2 step (d) part 1.

2. Generate 5-z-bin Gaussian covariance:

    i. Obtain unbinned Gaussian covariance blocks, using `gaussian_cov.get_cov_blocks`:

    ```python
    [python]

    n_zbin = 5
    she_she_input_filemask = 'path-to-theory-cls/shear_cl/bin_{hi_zbin}_{lo_zbin}.txt'
    lmax_in = 8000
    lmin_in = 2
    she_nl_path = 'path-to-shear-noise-power-spectrum-from-step-c.txt'
    noise_lmin = 2
    mask_path = 'path-to-mask.fits.gz'
    nside = 4906
    lmax_mix = lmax_in
    lmax_out = 5000
    lmin_out = 2
    save_path = 'path-to-save-block-{spec1_idx}_{spec2_idx}.npz'

    gaussian_cov.get_cov_blocks(n_zbin, she_she_input_filemask, lmax_in, lmin_in, she_nl_path, noise_lmin, mask_path, nside, lmax_mix, lmax_out, lmin_out, save_path)
    ```

    ii. Apply binning matrix to each block, using `post_processing.bin_blocks`:

    ```python
    [python]

    input_dir = 'path-to-gaussian-blocks-from-above'
    input_filemask = 'block-filename-[0-9]*_[0-9]*.npz'
    input_label = 'cov_block'
    binmat_path = 'path-to-binning-matrix-from-step-d.npz'
    output_dir = 'path-to-output-directory'

    post_processing.bin_blocks(input_dir, input_filemask, input_label, binmat_path, output_dir)
    ```

    iii. Combine blocks into a single matrix, using `post_processing.combine_blocks`:

    ```python
    [python]

    input_filemask = 'path-to-binned-blocks-from-above/block-filename-{spec1}_{spec2}.npz
    input_label = 'cov_binned'
    save_path = 'path-to-save-full-matrix.npz'
    n_spec = 15
    n_bp = 12

    post_processing.combine_blocks(input_filemask, input_label, save_path, n_spec, n_bp)
    ```

3. Generate 5-z-bin super-sample covariance:

    i. Calculate mask power spectrum - as Fig. 2 step (d) part 3.

    ii. Obtain unmixed unbinned super-sample covariance blocks - as Fig. 6 steps (c)-(d).

    iii. Apply mixing matrix to all blocks - as Fig. 6 step (f).

    iv. Apply binning matrix to all blocks - as for Cov_G above.

    v. Combine all blocks into a single matrix - as for Cov_G above.

4. Generate approximate 5-z-bin connected non-Gaussian covariance:

    i. Multiply the full-sky sparsely-sampled connected non-Gaussian covariance obtained in step (h) by 1/fsky to mimic the output from CosmoLike, using `cosmolike_utils.txt_to_npz`:

    ```python
    [python]

    input_filemask = 'path-to-cov_cng-blocks/cov_cng_spec1_[0-9]*_spec2_[0-9]*.txt'
    fsky = # 1 for full sky, .302 for full Euclid-like mask, .062 for Euclid DR1-like mask

    cosmolike_utils.txt_to_npz(input_filemask, fsky=fsky)
    ```

    ii. Apply binning approximation to each block, using `cng_approx.apply_bin_weights`:

    ```python
    [python]

    input_dir = 'path-to-cov_cng-blocks'
    output_dir = 'path-to-save-binned-cng-blocks'
    filemask = 'cov_cng_spec1_[0-9]*_spec2_[0-9]*.npz'
    weights_path = 'path-to-bin-weights-from-step-i.npz'

    cng_approx.apply_bin_weights(input_dir, output_dir, filemask, weights_path)
    ```

    iii. Obtain weights for mixing approximation, using `cng_approx.get_mix_weights`:

    ```python
    [python]

    full_cov_path = 'path-to-full-fullsky-cng-block-from-step-f.npz'
    binmat_path = 'path-to-bandpower-binning-matrix-from-step-d.npz'
    bin_weights_path = 'path-to-approx-binning-weights-from-step-i.npz'
    mixmat_path = 'path-to-mixing-matrix-from-step-j1.npz'
    lmin = 2
    save_path = 'path-to-save-mixing-weights.npz'
    fsky = # 1 for full sky, .302 for full Euclid-like mask, .062 for Euclid DR1-like mask

    cng_approx.get_mix_weights(full_cov_path, binmat_path, bin_weights_path, mixmat_path, lmin, save_path, fsky=fsky)
    ```

    iv. Apply mixing approximation to each block, using `cng_approx.apply_mix_weights`:

    ```python
    [python]

    input_dir = 'path-to-binned-cng-blocks-from-part-ii'
    output_dir = 'path-to-save-mixed-cng-blocks'
    filemask = 'cov_cng_spec1_[0-9]*_spec2_[0-9]*.npz'
    weights_path = 'path-to-mixing-weights-from-part-iii.npz'

    cng_approx.apply_mix_weights(input_dir, output_dir, filemask, weights_path)
    ```

    v. Combine all blocks into a single matrix - as for Cov_G above.

5. Form composite covariance matrices (G, G+SS, G+CNG, G+SS+CNG), using `post_processing.get_composite_covs`:

    ```python
    [python]

    cov_g_path = 'path-to-cov_g-from-step-j2.npz'
    cov_ss_path = 'path-to-cov_ss-from-step-j3.npz'
    cov_cng = 'path-to-cov_cng-from-step-j4.npz'
    output_path = 'path-to-save-cov_{label}.npz

    post_processing.get_composite_covs(cov_g_path, cov_ss_path, cov_cng_path, output_path)
    ```

6. Generate a mock observation with the total covariance, using `simulation.get_obs`:

    ```python
    [python]

    theory_cl_dir = 'path-to-fiducial-theory-dir-from-step-a/shear_cl'
    nl_path = 'path-to-shear-noise-power-spectrum-from-step-c.txt'
    mixmat_path = 'path-to-mixing-matrix-from-step-j1.npz'
    binmat_path = 'path-to-bandpower-binning-matrix-from-step-d.npz'
    cov_tot_path = 'path-to-total-covariance-from-step-j5.npz'
    n_zbin = 5
    lmax_in = 8000
    lmin_in = 2
    lmax_obs = 5000
    lmin_obs = 2
    n_bandpower = 12
    save_path = 'path-to-save-observed-bandpowers.npz'

    simulation.get_obs(theory_cl_dir, nl_path, mixmat_path, binmat_path, cov_tot_path, n_zbin, lmax_in, lmin_in, lmax_obs, lmin_obs, n_bandpower, save_path)
    ```

7. For each composite covariance (G, G+SS, G+CNG, G+SS+CNG):

    i. Evaluate the likelihood over the `w`-`wa` grid, using `likelihood.run_likelihood`:

    ```python
    [python]

    grid_dir = 'path-to-w0-wa-grid-from-step-a'
    varied_params = ['w', 'wa']
    save_path = 'path-to-save-likelihood.txt'
    obs_path = 'path-to-observation-from-step-j6.npz'
    n_zbin = 5
    cov_path = 'path-to-composite-covariance.npz'
    mixmat_path = 'path-to-mixing-matrix-from-step-j1.npz'
    binmat_path = 'path-to-bandpower-binning-matrix-from-step-d.npz'
    she_nl_path = 'path-to-shear-noise-power-spectrum-from-step-c.txt'
    noise_lmin = 2
    lmax_in = 8000
    lmin_in = 2
    lmax_like = 5000
    lmin_like = 2

    likelihood.run_likelihood(grid_dir, varied_params, save_path, obs_bp_path, n_zbin, cov_path, mixmat_path, binmat_path, she_nl_path, noise_lmin, lmax_in, lmin_in, lmax_like, lmin_like)
    ```

    ii. Evaluate the likelihood over the `omega_m`-`sigma_8` grid - as above for `w`-`wa`, except:

    ```python
    grid_dir = 'path-to-omm-si8-grid-from-step-b'
    varied_params = ['omega_m', 'sigma_8']
    ```

k) Plot all posteriors (for all masks and all combinations of covariance components) together, using `plotting.post_2d`:

```python
[python]

panels = [
    [ # Top row: w0-wa
        { # Full sky
            'like_paths': ['likelihood-full_sky-w0_wa-cov_tot.txt',
                           'likelihood-full_sky-w0_wa-cov_g_ss.txt',
                           'likelihood-full_sky-w0_wa-cov_g_cng.txt',
                           'likelihood-full_sky-w0_wa-cov_g.txt'],
            'xlims': (-1.111, -0.902),
            'ylims': (-0.314, 0.347),
            'smooth_sigma': [2.9, 2.6, 2.1, 1.5] # chosen to replicate unsmoothed rel. areas: 1, .82, .63, .38
        },
        { # Full Euclid-like
            'like_paths': ['likelihood-full_euclid-w0_wa-cov_tot.txt',
                           'likelihood-full_euclid-w0_wa-cov_g_ss.txt',
                           'likelihood-full_euclid-w0_wa-cov_g_cng.txt',
                           'likelihood-full_euclid-w0_wa-cov_g.txt'],
            'xlims': (-1.1514, -0.78),
            'ylims': (-0.717, 0.52),
            'smooth_sigma': [2.7, 2.5, 1.9, 1.4] # chosen to replicate unsmoothed rel. areas: 1, .84, .61, .36
        },
        { # Euclid DR1-like
            'like_paths': ['likelihood-dr1-w0_wa-cov_tot.txt',
                           'likelihood-dr1-w0_wa-cov_g_ss.txt',
                           'likelihood-dr1-w0_wa-cov_g_cng.txt',
                           'likelihood-dr1-w0_wa-cov_g.txt'],
            'xlims': (-1.459, -0.637),
            'ylims': (-1.185, 1.280),
            'smooth_sigma': [3.6, 3.3, 2.2, 1.5] # chosen to replicate unsmoothed rel. areas: 1, .85, .51, .30
        }
    ],
    [ # Bottom row: omm-si8
        { # Full sky
            'like_paths': ['likelihood-full_sky-omm_si8-cov_tot.txt',
                           'likelihood-full_sky-omm_si8-cov_g_ss.txt',
                           'likelihood-full_sky-omm_si8-cov_g_cng.txt',
                           'likelihood-full_sky-omm_si8-cov_g.txt'],
            'xlims': (0.30855, 0.31896),
            'ylims': (0.81936, 0.83384),
            'smooth_sigma': [1.8, 1.7, 1.3, 1] # chosen to replicate unsmoothed rel. areas: 1, .90, .70, .49
        },
        { # Full Euclid-like
            'like_paths': ['likelihood-full_euclid-omm_si8-cov_tot.txt',
                           'likelihood-full_euclid-omm_si8-cov_g_ss.txt',
                           'likelihood-full_euclid-omm_si8-cov_g_cng.txt',
                           'likelihood-full_euclid-omm_si8-cov_g.txt'],
            'xlims': (0.3053, 0.3244),
            'ylims': (0.8113, 0.8374),
            'smooth_sigma': [2.0, 1.8, 1.4, 1.0] # chosen to replicate unsmoothed rel. areas: 1, .90, .66, .45
        },
        { # Euclid DR1-like
            'like_paths': ['likelihood-dr1-omm_si8-cov_tot.txt',
                           'likelihood-dr1-omm_si8-cov_g_ss.txt',
                           'likelihood-dr1-omm_si8-cov_g_cng.txt',
                           'likelihood-dr1-omm_si8-cov_g.txt'],
            'xlims': (0.28474, 0.32884),
            'ylims': (0.8062, 0.8703),
            'smooth_sigma': [2.3, 2.2, 1.4, 1.0] # chosen to replicate unsmoothed rel. areas: 1, .92, .54, .37
        }
    ]
]
labels = ['G + SS + CNG', 'G + SS', 'G + CNG', 'G']
colours = [['C0'], ['C1'], ['C3'], ['C2']]
linestyles = [['-'], [(0, (5, 5))], ['dotted'], ['-.']]
contour_levels_sig = [1, 3]
column_titles = ['Full sky', 'Full Euclid-like', 'Euclid DR1-like']
param_labels = [(r'$w_0$', r'$w_a$'), (r'$\Omega_\mathrm{m}$', r'$\sigma_8$')]

plotting.post_2d(panels, labels, colours, linestyles, contour_levels_sig, column_titles, param_labels)
```

## Figure 8: 3D posteriors

a) Create three-dimensional theory grid of `w`, `wa`, `omega_m` - as Fig. 7 step (a) except:

1. In the input to [gaussian_cl_likelihood.python.cosmosis_utils.generate_chain_input](https://gaussian-cl-likelihood.readthedocs.io/en/latest/source/cosmosis_utils.html#cosmosis_utils.generate_chain_input), use the following parameter ranges:

    ```python
    params = {
        'cosmological_parameters-w': {
            'min': -1.23,
            'max': -0.70,
            'steps': 51
        },
        'cosmological_parameters-wa': {
            'min': -1.16,
            'max': 0.80,
            'steps': 51
        },
        'cosmological_parameters-omega_m': {
            'min': 0.3046,
            'max': 0.3230,
            'steps': 51
        }
    }
    ```

2. In the CosmoSIS values ini file, use a broad range for `omega_m` as well as for `w` and `wa`:

    ```ini
    w = -5.0 -1.0 0.0
    wa = -10.0 0.0 10.0
    omega_m = 0.1 0.3 0.9
    ```

b) As Fig. 7 steps (c)-(g) for the full Euclid-like mask only, with the following differences:

1. In the likelihood setup parameters for `likelihood.run_likelihood`, use:

    ```python
    [python]

    grid_dir = 'path-to-w0-wa-omm-grid-from-step-a'
    varied_params = ['w', 'wa', 'omega_m']
    ```

c) For each combination of covariance components (G, G+SS, G+CNG, G+SS+CNG), form 3D posterior grid using `plot_utils.get_3d_post`:

```python
[python]

log_like_paths = ['path-to-like-cov_g.txt', 'path-to-like-cov_g_ss.txt', 'path-to-like-cov_g_cng.txt', 'path-to-like-cov_tot.txt']
save_paths = ['path-to-save-post-cov_g.npz', 'path-to-save-post-cov_g_ss.npz', 'path-to-save-post-cov_g_cng.npz', 'path-to-save-post-cov_tot.npz']

for log_like_path, save_path in zip(log_like_paths, save_paths):
    plot_utils.get_3d_post(log_like_path, save_path)
```

d) Plot all posteriors on a single triangle plot, using `plotting.post_3d`:

```python
[python]

post_paths = ['path-to-post-cov_g.npz', 'path-to-save-post-cov_g_ss.npz', 'path-to-save-post-cov_g_cng.npz', 'path-to-save-post-cov_tot.npz']
labels = ['G + SS + CNG', 'G + SS', 'G + CNG', 'G']
colours = ['C0', 'C1', 'C3', 'C2']
linestyles = ['-', (0, (5, 5)), 'dotted', '-.']
contour_levels_sig = [1, 3]
x_label = r'$w_0$'
y_label = r'$w_a$'
z_label = r'$\Omega_\mathrm{m}$'
x_lims = (-1.19, -0.74)
y_lims = (-0.89, 0.57)
z_lims = (0.3053, 0.3222)
# Each set of smoothing values is chosen to preserve unsmoothed relative areas/widths
smooth_xy = [2.9, 2.8, 2.8, 2.7] # target rel. areas:  1, .89, .84, .69
smooth_xz = [3.2, 3.0, 2.6, 1.3] # target rel. areas:  1, .88, .75, .51
smooth_yz = [[2, 2.9], [1.9, 2.7], [1.7, 2.5], [1, 1.4]] # target rel. areas:  1, .89, .79, .57
smooth_x = [3.5, 3.1, 2.7, 2]  # target rel. widths: 1, .90, .83, .69
smooth_y = [3, 3.1, 2.8, 2.4]  # target rel. widths: 1, .96, .85, .74
smooth_z = [3.5, 3.2, 3.3, 3]  # target rel. widths: 1, .97, .94, .90

plotting.post_3d(post_paths, labels, colours, linestyles, contour_levels_sig, x_label=x_label, y_label=y_label, z_label=z_label, x_lims=x_lims, y_lims=y_lims, z_lims=z_lims, smooth_xy=smooth_xy, smooth_xz=smooth_xz, smooth_yz=smooth_yz, smooth_x=smooth_x, smooth_y=smooth_y, smooth_z=smooth_z)
```
