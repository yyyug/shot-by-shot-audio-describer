from processing.vlm_describer import describe_frames

def test_describe_frames_signature():
    # This will fail until implementation exists
    result = describe_frames([], "test-key")
    assert isinstance(result, str)
