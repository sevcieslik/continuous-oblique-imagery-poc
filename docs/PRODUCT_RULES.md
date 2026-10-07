# Product rules

## 1. Continuity is the product

The final product is the ordered series of legitimate acquisition frames covering a circuit from
start to finish. It is not "one image per span" and it is not a tower-photo catalogue.

A legal frame must be retained when it contributes to continuity, including frames that show:

- only part of a span;
- a single structure;
- no complete span;
- visually weak tower evidence but valid acquisition geometry.

Do not prune a frame solely because both endpoint structures are not visible.

## 2. Membership vs label

These are different decisions.

**Membership:** does this frame belong to a legitimate circuit pass?

**Reference:** what span or structure label should be written on the frame?

A low-confidence reference must not automatically remove a legal frame.

## 3. Pass direction vs view direction

Keep separate fields:

- `strip_direction` / `pass_direction`: `FORWARD_PASS`, `BACKWARD_PASS`, `CROSSING_PASS`, `UNKNOWN_PASS`;
- `view_direction`: `FORWARD`, `BACKWARD`, `SIDE`, `UNKNOWN`.

The product may legitimately contain two complete sequences: one travelling up-circuit and another
down-circuit. Continuity must be measured independently for each direction.

## 4. Crossing/side imagery

Crossing or side acquisition is retained in source/QA indexes but excluded from the canonical 163
sequence unless explicitly accepted. Footprint intersection with the circuit is not sufficient to
make a frame a legal product image.

## 5. DINOv3 is QA only

DINOv3 may flag weak overlap, possible sequence jumps, view changes or duplicate-like similarity.
It must not directly delete an image from the legal sequence.

## 6. Structure crops are downstream evidence

Structure detection/cropping is separate from 163 sequence construction. If no acceptable visual
structure detection is found, save no crop and retain the 163 frame.

All legitimate structure observations may be preserved; later FLI matching can choose or rank a
`best_view` without deleting alternate useful views.
