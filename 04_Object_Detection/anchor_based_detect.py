# You don't hand-implement anchors yourself when using a modern library -
# this is illustrative of the CONCEPT, not something you'll write from scratch.

# Example: a small set of anchor box shapes (width, height) in pixels,
# designed to cover different object shapes at one grid position
anchor_boxes = [
    (10, 13),   # small, roughly square (e.g. a distant small object)
    (30, 61),   # tall and narrow (e.g. a person)
    (116, 90),  # large and wide (e.g. a car seen from the side)
]

# At each grid cell, the model doesn't predict a box from nothing -
# it predicts: "which anchor is closest to what I see, and how do I 
# need to stretch/shift THAT anchor to fit the real object?"
# e.g. predicted_width = anchor_width * exp(predicted_offset)

print("Example anchors (w,h):", anchor_boxes)