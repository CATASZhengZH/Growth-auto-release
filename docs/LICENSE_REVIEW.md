# License Review

## Draft Status

This workspace is not ready for public release until the project license is
approved by the copyright holder or authorized institution. The placeholder
root `LICENSE` grants no project-level rights.

## Technical Finding

The release uses Ultralytics 8.3.226 and distributes a checkpoint whose embedded
metadata identifies AGPL-3.0. Ultralytics states that code and models used under
its open-source terms are subject to AGPL-3.0 unless an applicable enterprise
license is held. For that reason, an MIT-only release is not supported by the
current evidence.

If no enterprise license or separate written permission applies, AGPL-3.0 is
the conservative project-license option. The authors and institution must also
confirm that the original model module, Growth-auto code, checkpoint, and five
field images may be redistributed under the selected terms.

## Human Decisions Required

1. Confirm whether an Ultralytics enterprise license applies.
2. Confirm ownership of the original source code and checkpoint.
3. Confirm permission to redistribute the five demo images.
4. Select and insert the final project license.
5. Confirm whether institutional policy requires additional notices.

See `THIRD_PARTY_NOTICES.md` for the dependency inventory. This is a technical
release review, not legal advice.
