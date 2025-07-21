from torch_fidelity import calculate_metrics

metrics = calculate_metrics(
    input1='path_to_real_images',
    input2='path_to_generated_images',
    cuda=True,
    isc=False,
    fid=True
)

print("FID score:", metrics['frechet_inception_distance'])
