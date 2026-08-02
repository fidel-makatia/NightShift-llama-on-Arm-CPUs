| model                          |       size |     params | backend    | threads |            test |                  t/s |
| ------------------------------ | ---------: | ---------: | ---------- | ------: | --------------: | -------------------: |
| llama4 17Bx128E (Maverick) Q2_K - Medium | 142.16 GiB |   400.71 B | CPU        |      48 |           pp512 |         69.77 ± 4.48 |
| llama4 17Bx128E (Maverick) Q2_K - Medium | 142.16 GiB |   400.71 B | CPU        |      48 |           tg128 |         21.75 ± 0.27 |
| llama4 17Bx128E (Maverick) Q2_K - Medium | 142.16 GiB |   400.71 B | CPU        |      64 |           pp512 |         92.77 ± 0.29 |
| llama4 17Bx128E (Maverick) Q2_K - Medium | 142.16 GiB |   400.71 B | CPU        |      64 |           tg128 |         21.96 ± 0.14 |
| llama4 17Bx128E (Maverick) Q2_K - Medium | 142.16 GiB |   400.71 B | CPU        |      96 |           pp512 |        133.32 ± 0.46 |
| llama4 17Bx128E (Maverick) Q2_K - Medium | 142.16 GiB |   400.71 B | CPU        |      96 |           tg128 |         17.22 ± 0.49 |

build: 876a432 (1)
