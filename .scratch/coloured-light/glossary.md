# Pending glossary: coloured light

Terms this effort adds to or changes in `GLOSSARY.md`. `GLOSSARY.md` stays as it is until sub-grids is merged. Then each entry below goes in as written, under the Light section, and this file goes.

## New terms

**Light colour**:
The colour of the voxel light an emissive voxel gives off, declared with its material. It's white unless declared.
_Avoid_: tint (what a translucent voxel will do to light passing through it), hue

## Reserved

- **Tint**: kept, with no entry yet, for what a translucent voxel will do to light passing through it, which comes with translucent voxels after this effort. Don't spend the word here.

## Open

- **Channel**: today it means sky light or voxel light ("one level per channel" in *Slot*, *Tag* and *Light precision*, "one channel at a time" in *Kernel*). If voxel light is held as red, green and blue, each would be another run of the kernel, which the kernel docs already call another channel. [The representation of coloured voxel light](issues/05-the-representation-of-coloured-voxel-light.md) settles the word.
