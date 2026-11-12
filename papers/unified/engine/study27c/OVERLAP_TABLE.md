# Region / context overlap table (Study 27b world; maintained for 27b/27c)

Strands of the witness braid: their contexts, which variables they share, whether they see the decision quantity g (contain M
and Y), whether they are hexagon neighbours (co-revision edges), and the region of the moment polytope their table occupies
(smallest expected cell count: interior >= 30, boundary layer 5-30, near face < 5) under the truth P and the far-side law Q.

| strand | context | n | sees g | hexagon neighbours | min cell P | min cell Q | region (P) | region (Q) |
|---|---|---|---|---|---|---|---|---|
| S0 | {M,C} | 6000 | no | S1, S3, S5 | 0.0963 | 0.0963 | interior | interior |
| S1 | {M,R} | 6000 | no | S0, S2 | 0.1177 | 0.1177 | interior | interior |
| S2 | {C,R} | 6000 | no | S1, S3 | 0.0928 | 0.0928 | interior | interior |
| S3 | {M,C,Y} | 6000 | yes | S0, S2, S4 | 0.0348 | 0.0300 | interior | interior |
| S4 | {M,H} | 6000 | no | S3, S5 | 0.1427 | 0.1427 | interior | interior |
| S5 | {R,H,Y} | 6000 | no | S0, S4 | 0.0451 | 0.0451 | interior | interior |
| MRY | {M,R,Y} | 3000 | yes | - | 0.0477 | 0.0402 | interior | interior |
| MHY | {M,H,Y} | 3000 | yes | - | 0.0681 | 0.0534 | interior | interior |
| I3000 | {M,Y} | 3000 | yes | - | 0.1553 | 0.1308 | interior | interior |
| I120k | {M,Y} | 120000 | yes | - | 0.1553 | 0.1308 | interior | interior |

Shared variables between contexts (overlap = the restriction maps of the sheaf):

| | S0 | S1 | S2 | S3 | S4 | S5 | MRY | MHY | I3000 | I120k |
|---|---|---|---|---|---|---|---|---|---|---|
| S0 | MC | M | C | MC | M | - | M | M | M | M |
| S1 | M | MR | R | M | M | R | MR | M | M | M |
| S2 | C | R | CR | C | - | R | R | - | - | - |
| S3 | MC | M | C | MCY | M | Y | MY | MY | MY | MY |
| S4 | M | M | - | M | MH | H | M | MH | M | M |
| S5 | - | R | R | Y | H | RHY | RY | HY | Y | Y |
| MRY | M | MR | R | MY | M | RY | MRY | MY | MY | MY |
| MHY | M | M | - | MY | MH | HY | MY | MHY | MY | MY |
| I3000 | M | M | - | MY | M | Y | MY | MY | MY | MY |
| I120k | M | M | - | MY | M | Y | MY | MY | MY | MY |
