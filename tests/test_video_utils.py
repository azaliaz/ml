from video_utils import (
    VideoUploadParams,
    parse_frame_step,
    source_frame_index,
    ExtractedFrame,
    patch_coco_archive_for_cvat_video,
    resolve_cvat_frame_list,
)


def test_parse_frame_step_defaults_to_one() -> None:
    assert parse_frame_step("") == 1
    assert parse_frame_step(None) == 1


def test_parse_frame_step_reads_step_value() -> None:
    assert parse_frame_step("step=5") == 5
    assert parse_frame_step("step=1,other=x") == 1


def test_video_upload_params_computes_stop_frame_from_max_frames() -> None:
    params = VideoUploadParams(frame_step=5, start_frame=0, max_frames=10).normalized(video_frame_count=10_000)
    assert params.stop_frame == 45


def test_video_upload_params_clamps_to_video_length() -> None:
    params = VideoUploadParams(frame_step=1, start_frame=0, max_frames=500).normalized(video_frame_count=100)
    assert params.stop_frame == 99


def test_source_frame_index() -> None:
    assert source_frame_index(0, start_frame=10, frame_step=5) == 10
    assert source_frame_index(3, start_frame=10, frame_step=5) == 25


def test_resolve_cvat_frame_list_expands_video_placeholder() -> None:
    meta = {
        "size": 3,
        "frames": [{"name": "clip.mp4", "width": 640, "height": 480}],
    }
    frames = resolve_cvat_frame_list(meta)
    assert len(frames) == 3
    assert frames[0]["name"] == "000000.jpg"
    assert frames[2]["name"] == "000002.jpg"


def test_patch_coco_archive_for_cvat_video(tmp_path) -> None:
    import json
    import zipfile

    coco = {
        "images": [{"id": 1, "file_name": "000001", "width": 100, "height": 100}],
        "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10]}],
        "categories": [{"id": 1, "name": "car"}],
    }
    archive = tmp_path / "preann.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        zf.writestr("annotations/annotations_coco.json", json.dumps(coco))

    extracted = [ExtractedFrame(local_path=str(tmp_path / "000001"), cvat_name="frame_000001.jpg", task_frame_index=0)]
    stats = patch_coco_archive_for_cvat_video(archive, extracted)
    assert stats["matched_images"] == 1

    with zipfile.ZipFile(archive, "r") as zf:
        patched = json.loads(zf.read("annotations/instances_default.json"))
    assert patched["images"][0]["file_name"] == "frame_000001.jpg"
    assert patched["images"][0]["id"] == 1
