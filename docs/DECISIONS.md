# Key design decisions

## D001 - Never use full-span visibility as a membership gate

**Status:** accepted.

Reason: review showed that partial-span/no-complete-span images can be required to maintain visual
continuity. This reversed the temporary v0.6 rule and became the v0.6.1+ product definition.

## D002 - Separate acquisition direction from target view direction

**Status:** accepted.

`pass_direction` answers how the aircraft traversed the circuit. `view_direction` answers how the camera
sits relative to the target/local tangent. They must not be collapsed into one field.

## D003 - Evaluate continuity per legitimate direction

**Status:** accepted.

Forward and backward coverage/gaps are assessed separately. Aggregate coverage can mask a missing
sequence in one direction.

## D004 - DINOv3 is advisory QA

**Status:** accepted.

DINO may flag transitions but cannot remove a legal frame. High local similarity is expected in an
overlapping acquisition sequence and is not itself duplicate evidence.

## D005 - Existing/manual 163 is benchmark-only

**Status:** accepted.

Manual work is audited before calibration. Ground truth is an adjudicated subset, not the complete
historical deliverable.

## D006 - Fail closed on structure crops

**Status:** accepted.

When visual structure evidence is insufficient, save no crop. Never save a geometry-only field/shadow
crop merely to produce an output. Crop failure never changes continuous-sequence membership.

## D007 - Geometry constrains detector search, not final visual truth

**Status:** accepted in v0.14.

Projection errors made pure geometry cropping unreliable; full-image visual detection was too noisy.
The combined solution is a geometry-derived local ROI followed by Grounding DINO and geometry-gated
assignment.

## D008 - FLI comparison is downstream

**Status:** accepted.

163 sequence construction should not be tightly coupled to later FLI matching. The pipeline exports
structure observations/crops; a separate module can rank or compare those views later.
