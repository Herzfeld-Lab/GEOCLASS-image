# Jakobshavn 2018-2021 10-Class Label Guide

Use visual geometry as the primary rule. Process interpretation can inform the notes, but it should not override what is visible in the split image.

Leave a patch unlabeled when it contains multiple meaningful patterns with no clear dominant geometry, is mostly cloud/shadow/nodata, is outside the glacier surface of interest, or cannot be labeled confidently from the patch alone.

## Classes

| ID | Class name | Visual rule | Include examples | Exclude examples |
| --- | --- | --- | --- | --- |
| 0 | Smooth or no visible crevasses | Little to no resolved crevasse structure; surface appears smooth, snow covered, or weakly textured. | Smooth snow/ice, low-texture areas, faint texture without clear lineation. | Cloud/shadow/nodata; patches with clear crevasse geometry. |
| 1 | Sparse isolated crevasses | A few separated crevasses with large unbroken areas between them. | Single fractures, short isolated cracks, low-density crevasses without a shared fabric. | Dense parallel sets; transition zones with several competing fabrics. |
| 2 | Thin parallel crevasses | Many narrow, similarly oriented crevasses with consistent spacing. | Fine lineated fields, narrow parallel cracks, subtle organized texture. | Wide/deep parallel crevasses; crosscutting sets. |
| 3 | Strong parallel crevasses | Prominent, wide, dark, or high-contrast crevasses sharing one dominant orientation. | Strong transverse/longitudinal bands, high-contrast parallel fracture trains. | Thin low-contrast lineations; chaotic fractured zones. |
| 4 | Curved or arcuate crevasses | Crevasses form curved, bowed, crescent, or fan-like traces with a coherent curvature. | Arcuate bands, curved transverse sets, fan-shaped crevasse fields. | Straight parallel sets; en echelon arrays without strong curvature. |
| 5 | En echelon crevasses | Short, offset, staggered cracks arranged in a consistent stepping pattern. | Ladder-like or overlapping short segments, oblique repeated offsets. | Continuous parallel bands; random short cracks. |
| 6 | Shear-margin oblique crevasses | Oblique, elongated, or banded crevasses consistent with shear-zone fabric; usually organized along a margin or boundary. | Diagonal shear bands, margin-parallel/oblique fractures, elongated shear textures. | Generic parallel crevasses away from a shear fabric; chaotic break-up. |
| 7 | Crosscutting multidirectional crevasses | Two or more clear crevasse orientations intersect or overprint each other, but the patch remains organized enough to label. | Orthogonal sets, intersecting parallel fabrics, overprinted crevasse directions. | Fully chaotic rubble-like texture; mixed patches where one class cannot be chosen confidently. |
| 8 | Chaotic heavily fractured crevasses | Dense fracture network with no single dominant orientation or repeated spacing. | Jumbled fracture fields, shattered texture, dense irregular crevasse networks. | Ordered crosscutting sets; mixed edge patches with large smooth areas. |
| 9 | Crevasses obscured by snow or low contrast | Crevasse pattern is visible but muted or partly buried; the defining feature is obscuration, not absence. | Snow-bridged crevasse traces, low-contrast buried lineations, visible-but-muted patterns. | Truly smooth/no-crevasse surfaces; cloud/shadow/nodata. |

## Mixed-Patch Rule

Do not force labels for mixed patches during the initial training set. If two or more classes occupy substantial parts of the split image and no single visual geometry dominates, leave the patch unlabeled and move on.

## Sampling Target

Start with about 75 labeled examples per class, balanced across years where possible. Add more labels later for classes with poor held-out-year performance, high confusion, or visibly unstable map predictions.
