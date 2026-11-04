from experiments.staged_reconstruction.RubberDuck_visibility_audit.qualification_repair import centered_intrinsics


def test_native_pixel_center_even_and_odd():
    for width,height in [(64,36),(65,37)]:
        k=centered_intrinsics(width,height)
        assert k[0,2]==(width-1)/2 and k[1,2]==(height-1)/2
        assert k[0,0]==100 and k[1,1]==100
