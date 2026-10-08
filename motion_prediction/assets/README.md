# Detector assets

`haarcascade_fullbody.xml` is the original full body detector previously stored in the repository root. It is included as package data so Haar detection works after installation from any working directory.

Its embedded notice credits Hannes Kruppa and Bernt Schiele, ETH Zurich, copyright 2004. The original copyright, redistribution conditions, and disclaimer are retained verbatim in the XML. These terms apply to the model separately from the project's MIT code license.

YOLOv3 configuration and weights are external assets. Supply their paths explicitly with `--config` and `--weights`. The original `coco.names` at the repository root remains a class order reference; person detection uses COCO class index zero and does not load this file.
