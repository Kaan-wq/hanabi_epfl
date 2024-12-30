# Game State 1

## Game State

```
Information Tokens: 8.0
Life Tokens: 3.0
Fireworks: R0 Y0 G0 W0 B0
Deck size: 40.0
Discards:
None

Hands:
Your hand:
XX || RYGWB12345
XX || RYGWB12345
XX || RYGWB12345
XX || RYGWB12345
XX || RYGWB12345

Other player's hand:
B3 R2 G2 R4 R4
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| Reveal R to P1 | 1.000 |  1.000  |    ★     |

Value: 0.920

---

# Game State 2

## Game State

```
Information Tokens: 7.0
Life Tokens: 3.0
Fireworks: R0 Y0 G0 W0 B0
Deck size: 40.0
Discards:
None

Hands:
Your hand:
XX || YGWB12345
RX || R12345
XX || YGWB12345
RX || R12345
RX || R12345

Other player's hand:
W2 G1 W2 Y5 W1
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 1 | 0.010 |  0.390  |    R     |
| Reveal W to P1 | 0.005 |  0.228  |          |
| Reveal 1 to P1 | 0.985 |  0.382  |    M     |

Value: 0.920

---

# Game State 3

## Game State

```
Information Tokens: 6.0
Life Tokens: 3.0
Fireworks: R0 Y0 G0 W0 B0
Deck size: 40.0
Discards:
None

Hands:
Your hand:
XX || RYGWB2345
X1 || RYGWB1
XX || RYGWB2345
XX || RYGWB2345
X1 || RYGWB1

Other player's hand:
B3 R2 G2 R4 R4
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 2 | 0.003 |  0.344  |    R     |
|  PLAY card 2   | 0.995 |  0.334  |    M     |
| Reveal 2 to P1 | 0.003 |  0.323  |          |

Value: 0.920

---

# Game State 4

## Game State

```
Information Tokens: 6.0
Life Tokens: 3.0
Fireworks: R0 Y0 G1 W0 B0
Deck size: 39.0
Discards:
None

Hands:
Your hand:
XX || YGWB12345
RX || R12345
XX || YGWB12345
RX || R12345
RX || R12345

Other player's hand:
W2 W2 Y5 W1 W3
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 1 | 0.990 |  0.606  |    ★     |
| Reveal W to P1 | 0.010 |  0.394  |          |

Value: 0.920

---

# Game State 5

## Game State

```
Information Tokens: 7.0
Life Tokens: 3.0
Fireworks: R0 Y0 G1 W0 B0
Deck size: 38.0
Discards:
B3

Hands:
Your hand:
XX || RYGWB2345
XX || RYGWB2345
XX || RYGWB2345
X1 || RYGWB1
XX || RYGWB12345

Other player's hand:
R2 G2 R4 R4 G4
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 4 | 0.003 |  0.288  |    R     |
|  PLAY card 4   | 0.992 |  0.270  |    M     |
| Reveal 2 to P1 | 0.003 |  0.276  |          |
| Reveal 4 to P1 | 0.003 |  0.166  |          |

Value: 0.920

---

# Game State 6

## Game State

```
Information Tokens: 7.0
Life Tokens: 3.0
Fireworks: R0 Y0 G1 W1 B0
Deck size: 37.0
Discards:
B3

Hands:
Your hand:
RX || R12345
XX || YGWB12345
RX || R12345
RX || R12345
XX || RYGWB12345

Other player's hand:
W2 W2 Y5 W3 R1
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 2 | 0.043 |  0.356  |          |
| Reveal W to P1 | 0.028 |  0.241  |          |
| Reveal 2 to P1 | 0.930 |  0.403  |    ★     |

Value: 0.920

---

# Game State 7

## Game State

```
Information Tokens: 6.0
Life Tokens: 3.0
Fireworks: R0 Y0 G1 W1 B0
Deck size: 37.0
Discards:
B3

Hands:
Your hand:
X2 || RYGWB2
X2 || RYGWB2
XX || RYGWB345
XX || RYGWB1345
XX || RYGWB1345

Other player's hand:
R2 G2 R4 R4 G4
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 4 | 0.038 |  0.420  |    R     |
| Reveal 2 to P1 | 0.033 |  0.362  |          |
| Reveal 4 to P1 | 0.930 |  0.219  |    M     |

Value: 0.920

---

# Game State 8

## Game State

```
Information Tokens: 5.0
Life Tokens: 3.0
Fireworks: R0 Y0 G1 W1 B0
Deck size: 37.0
Discards:
B3

Hands:
Your hand:
RX || R1235
XX || YGWB1235
R4 || R4
R4 || R4
X4 || RYGWB4

Other player's hand:
W2 W2 Y5 W3 R1
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 2 | 0.008 |  0.591  |    R     |
| Reveal W to P1 | 0.992 |  0.409  |    M     |

Value: 0.920

---

# Game State 9

## Game State

```
Information Tokens: 4.0
Life Tokens: 3.0
Fireworks: R0 Y0 G1 W1 B0
Deck size: 37.0
Discards:
B3

Hands:
Your hand:
W2 || W2
W2 || W2
XX || RYGB345
WX || W1345
XX || RYGB1345

Other player's hand:
R2 G2 R4 R4 G4
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 4 | 0.003 |  0.348  |    R     |
|  PLAY card 1   | 0.995 |  0.334  |    M     |
| Reveal 2 to P1 | 0.003 |  0.318  |          |

Value: 0.920

---

# Game State 10

## Game State

```
Information Tokens: 4.0
Life Tokens: 3.0
Fireworks: R0 Y0 G1 W2 B0
Deck size: 36.0
Discards:
B3

Hands:
Your hand:
RX || R1235
XX || YGWB1235
R4 || R4
R4 || R4
X4 || RYGWB4

Other player's hand:
W2 Y5 W3 R1 R3
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 2 | 0.010 |  0.605  |    R     |
| Reveal 3 to P1 | 0.990 |  0.395  |    M     |

Value: 0.920

---

# Game State 11

## Game State

```
Information Tokens: 3.0
Life Tokens: 3.0
Fireworks: R0 Y0 G1 W2 B0
Deck size: 36.0
Discards:
B3

Hands:
Your hand:
W2 || W2
XX || RYGB45
W3 || W3
XX || RYGB145
X3 || RYGWB3

Other player's hand:
R2 G2 R4 R4 G4
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 1 | 0.003 |  0.360  |    R     |
|  PLAY card 3   | 0.995 |  0.349  |    M     |
| Reveal 2 to P1 | 0.003 |  0.291  |          |

Value: 0.920

---

# Game State 12

## Game State

```
Information Tokens: 3.0
Life Tokens: 3.0
Fireworks: R0 Y0 G1 W3 B0
Deck size: 35.0
Discards:
B3

Hands:
Your hand:
RX || R1235
XX || YGWB1235
R4 || R4
R4 || R4
X4 || RYGWB4

Other player's hand:
W2 Y5 R1 R3 W3
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 2 | 0.008 |  0.345  |          |
| Reveal R to P1 | 0.013 |  0.281  |          |
| Reveal 1 to P1 | 0.980 |  0.374  |    ★     |

Value: 0.920

---

# Game State 13

## Game State

```
Information Tokens: 2.0
Life Tokens: 3.0
Fireworks: R0 Y0 G1 W3 B0
Deck size: 35.0
Discards:
B3

Hands:
Your hand:
W2 || W2
XX || RYGB45
X1 || RYGB1
X3 || RYGWB3
XX || RYGWB2345

Other player's hand:
R2 G2 R4 R4 G4
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 1 | 0.005 |  0.330  |          |
|  PLAY card 3   | 0.990 |  0.350  |    ★     |
| Reveal 2 to P1 | 0.005 |  0.319  |          |

Value: 0.920

---

# Game State 14

## Game State

```
Information Tokens: 2.0
Life Tokens: 3.0
Fireworks: R1 Y0 G1 W3 B0
Deck size: 34.0
Discards:
B3

Hands:
Your hand:
RX || R1235
XX || YGWB1235
R4 || R4
R4 || R4
X4 || RYGWB4

Other player's hand:
W2 Y5 R3 W3 Y2
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 1 | 0.025 |  0.390  |    R     |
| Reveal R to P1 | 0.957 |  0.341  |    M     |
| Reveal Y to P1 | 0.018 |  0.268  |          |

Value: 0.920

---

# Game State 15

## Game State

```
Information Tokens: 1.0
Life Tokens: 3.0
Fireworks: R1 Y0 G1 W3 B0
Deck size: 34.0
Discards:
B3

Hands:
Your hand:
W2 || W2
XX || YGB45
R3 || R3
XX || YGWB2345
XX || YGWB12345

Other player's hand:
R2 G2 R4 R4 G4
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 1 | 0.005 |  0.366  |    R     |
| Reveal G to P1 | 0.003 |  0.278  |          |
| Reveal 2 to P1 | 0.992 |  0.356  |    M     |

Value: 0.920

---

# Game State 16

## Game State

```
Information Tokens: 0.0
Life Tokens: 3.0
Fireworks: R1 Y0 G1 W3 B0
Deck size: 34.0
Discards:
B3

Hands:
Your hand:
R2 || R2
X2 || YGWB2
R4 || R4
R4 || R4
X4 || RYGWB4

Other player's hand:
W2 Y5 R3 W3 Y2
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 1 | 0.003 |  0.514  |    R     |
|  PLAY card 1   | 0.997 |  0.486  |    M     |

Value: 0.920

---

# Game State 17

## Game State

```
Information Tokens: 0.0
Life Tokens: 3.0
Fireworks: R2 Y0 G1 W3 B0
Deck size: 33.0
Discards:
B3

Hands:
Your hand:
W2 || W2
XX || YGB45
R3 || R3
XX || YGWB2345
XX || YGWB12345

Other player's hand:
G2 R4 R4 G4 G3
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 1 | 0.003 |  0.491  |          |
|  PLAY card 3   | 0.997 |  0.509  |    ★     |

Value: 0.920

---

# Game State 18

## Game State

```
Information Tokens: 0.0
Life Tokens: 3.0
Fireworks: R3 Y0 G1 W3 B0
Deck size: 32.0
Discards:
B3

Hands:
Your hand:
X2 || YGWB2
R4 || R4
R4 || R4
X4 || RYGWB4
XX || RYGWB12345

Other player's hand:
W2 Y5 W3 Y2 G4
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 5 | 0.005 |  0.530  |    R     |
|  PLAY card 2   | 0.995 |  0.470  |    M     |

Value: 0.920

---

# Game State 19

## Game State

```
Information Tokens: 0.0
Life Tokens: 3.0
Fireworks: R4 Y0 G1 W3 B0
Deck size: 31.0
Discards:
B3

Hands:
Your hand:
W2 || W2
XX || YGB45
XX || YGWB2345
XX || YGWB12345
XX || RYGWB12345

Other player's hand:
G2 R4 G4 G3 G1
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 1 | 1.000 |  1.000  |    ★     |

Value: 0.920

---

# Game State 20

## Game State

```
Information Tokens: 1.0
Life Tokens: 3.0
Fireworks: R4 Y0 G1 W3 B0
Deck size: 30.0
Discards:
W2 B3

Hands:
Your hand:
X2 || YGWB2
R4 || R4
X4 || RYGWB4
XX || RYGWB12345
XX || RYGWB12345

Other player's hand:
Y5 W3 Y2 G4 Y2
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 2 | 0.942 |  0.565  |    ★     |
| Reveal Y to P1 | 0.058 |  0.435  |          |

Value: 0.920

---

# Game State 21

## Game State

```
Information Tokens: 2.0
Life Tokens: 3.0
Fireworks: R4 Y0 G1 W3 B0
Deck size: 29.0
Discards:
R4 W2 B3

Hands:
Your hand:
XX || YGB45
XX || YGWB2345
XX || YGWB12345
XX || RYGWB12345
XX || RYGWB12345

Other player's hand:
G2 G4 G3 G1 W5
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 4 | 0.013 |  0.575  |    R     |
| Reveal G to P1 | 0.987 |  0.425  |    M     |

Value: 0.920

---

# Game State 22

## Game State

```
Information Tokens: 1.0
Life Tokens: 3.0
Fireworks: R4 Y0 G1 W3 B0
Deck size: 29.0
Discards:
R4 W2 B3

Hands:
Your hand:
G2 || G2
G4 || G4
GX || G12345
GX || G12345
XX || RYWB12345

Other player's hand:
Y5 W3 Y2 G4 Y2
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 3 | 0.003 |  0.323  |          |
|  PLAY card 1   | 0.995 |  0.388  |    ★     |
| Reveal Y to P1 | 0.003 |  0.289  |          |

Value: 0.920

---

# Game State 23

## Game State

```
Information Tokens: 1.0
Life Tokens: 3.0
Fireworks: R4 Y0 G2 W3 B0
Deck size: 28.0
Discards:
R4 W2 B3

Hands:
Your hand:
XX || YGB45
XX || YGWB2345
XX || YGWB12345
XX || RYGWB12345
XX || RYGWB12345

Other player's hand:
G4 G3 G1 W5 W1
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 4 | 0.005 |  0.351  |          |
| Reveal 1 to P1 | 0.008 |  0.395  |    R     |
| Reveal 3 to P1 | 0.987 |  0.254  |    M     |

Value: 0.920

---

# Game State 24

## Game State

```
Information Tokens: 0.0
Life Tokens: 3.0
Fireworks: R4 Y0 G2 W3 B0
Deck size: 28.0
Discards:
R4 W2 B3

Hands:
Your hand:
G4 || G4
G3 || G3
GX || G1245
XX || RYWB1245
XX || RYGWB1245

Other player's hand:
Y5 W3 Y2 G4 Y2
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 3 | 0.003 |  0.494  |          |
|  PLAY card 2   | 0.997 |  0.506  |    ★     |

Value: 0.920

---

# Game State 25

## Game State

```
Information Tokens: 0.0
Life Tokens: 3.0
Fireworks: R4 Y0 G3 W3 B0
Deck size: 27.0
Discards:
R4 W2 B3

Hands:
Your hand:
XX || YGB45
XX || YGWB2345
XX || YGWB12345
XX || RYGWB12345
XX || RYGWB12345

Other player's hand:
G4 G1 W5 W1 B5
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 4 | 1.000 |  1.000  |    ★     |

Value: 0.920

---

# Game State 26

## Game State

```
Information Tokens: 1.0
Life Tokens: 3.0
Fireworks: R4 Y0 G3 W3 B0
Deck size: 26.0
Discards:
R4 G4 W2 B3

Hands:
Your hand:
G4 || G4
GX || G1245
XX || RYWB1245
XX || RYGWB1245
XX || RYGWB12345

Other player's hand:
Y5 W3 Y2 Y2 B2
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 2 | 0.003 |  0.392  |    R     |
|  PLAY card 1   | 0.995 |  0.350  |    M     |
| Reveal Y to P1 | 0.003 |  0.258  |          |

Value: 0.920

---

# Game State 27

## Game State

```
Information Tokens: 1.0
Life Tokens: 3.0
Fireworks: R4 Y0 G4 W3 B0
Deck size: 25.0
Discards:
R4 G4 W2 B3

Hands:
Your hand:
XX || YGB45
XX || YGWB2345
XX || YGWB12345
XX || RYGWB12345
XX || RYGWB12345

Other player's hand:
G1 W5 W1 B5 B3
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 4 | 0.165 |  0.470  |          |
| Reveal 1 to P1 | 0.835 |  0.530  |    ★     |

Value: 0.920

---

# Game State 28

## Game State

```
Information Tokens: 0.0
Life Tokens: 3.0
Fireworks: R4 Y0 G4 W3 B0
Deck size: 25.0
Discards:
R4 G4 W2 B3

Hands:
Your hand:
G1 || G1
XX || RYWB245
X1 || RYGWB1
XX || RYGWB2345
XX || RYGWB2345

Other player's hand:
Y5 W3 Y2 Y2 B2
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 1 | 1.000 |  1.000  |    ★     |

Value: 0.920

---

# Game State 29

## Game State

```
Information Tokens: 1.0
Life Tokens: 3.0
Fireworks: R4 Y0 G4 W3 B0
Deck size: 24.0
Discards:
R4 G1 G4 W2 B3

Hands:
Your hand:
XX || YGB45
XX || YGWB2345
XX || YGWB12345
XX || RYGWB12345
XX || RYGWB12345

Other player's hand:
W5 W1 B5 B3 R1
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 4 | 0.822 |  0.634  |    ★     |
| Reveal W to P1 | 0.178 |  0.366  |          |

Value: 0.920

---

# Game State 30

## Game State

```
Information Tokens: 2.0
Life Tokens: 3.0
Fireworks: R4 Y0 G4 W3 B0
Deck size: 23.0
Discards:
R4 Y2 G1 G4 W2 B3

Hands:
Your hand:
XX || RYWB245
X1 || RYGWB1
XX || RYGWB2345
XX || RYGWB2345
XX || RYGWB12345

Other player's hand:
Y5 W3 Y2 B2 R1
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 2 | 0.068 |  0.579  |    R     |
| Reveal Y to P1 | 0.932 |  0.421  |    M     |

Value: 0.920

---

# Game State 31

## Game State

```
Information Tokens: 1.0
Life Tokens: 3.0
Fireworks: R4 Y0 G4 W3 B0
Deck size: 23.0
Discards:
R4 Y2 G1 G4 W2 B3

Hands:
Your hand:
YX || Y45
XX || GWB2345
YX || Y12345
XX || RGWB12345
XX || RGWB12345

Other player's hand:
W5 W1 B5 B3 R1
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 4 | 0.143 |  0.656  |    R     |
| Reveal W to P1 | 0.857 |  0.344  |    M     |

Value: 0.920

---

# Game State 32

## Game State

```
Information Tokens: 0.0
Life Tokens: 3.0
Fireworks: R4 Y0 G4 W3 B0
Deck size: 23.0
Discards:
R4 Y2 G1 G4 W2 B3

Hands:
Your hand:
WX || W245
W1 || W1
XX || RYGB2345
XX || RYGB2345
XX || RYGB12345

Other player's hand:
Y5 W3 Y2 B2 R1
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 2 | 1.000 |  1.000  |    ★     |

Value: 0.920

---

# Game State 33

## Game State

```
Information Tokens: 1.0
Life Tokens: 3.0
Fireworks: R4 Y0 G4 W3 B0
Deck size: 22.0
Discards:
R4 Y2 G1 G4 W1 W2 B3

Hands:
Your hand:
YX || Y45
XX || GWB2345
YX || Y12345
XX || RGWB12345
XX || RGWB12345

Other player's hand:
W5 B5 B3 R1 W4
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 4 | 0.962 |  0.458  |    ★     |
| Reveal 4 to P1 | 0.018 |  0.259  |          |
| Reveal 5 to P1 | 0.020 |  0.282  |          |

Value: 0.920

---

# Game State 34

## Game State

```
Information Tokens: 2.0
Life Tokens: 3.0
Fireworks: R4 Y0 G4 W3 B0
Deck size: 21.0
Discards:
R4 Y2 G1 G4 W1 W2 B2 B3

Hands:
Your hand:
WX || W245
XX || RYGB2345
XX || RYGB2345
XX || RYGB12345
XX || RYGWB12345

Other player's hand:
Y5 W3 Y2 R1 W1
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 2 | 0.033 |  0.459  |    R     |
| Reveal W to P1 | 0.018 |  0.260  |          |
| Reveal 5 to P1 | 0.950 |  0.282  |    M     |

Value: 0.920

---

# Game State 35

## Game State

```
Information Tokens: 1.0
Life Tokens: 3.0
Fireworks: R4 Y0 G4 W3 B0
Deck size: 21.0
Discards:
R4 Y2 G1 G4 W1 W2 B2 B3

Hands:
Your hand:
Y5 || Y5
XX || GWB234
YX || Y1234
XX || RGWB1234
XX || RYGWB1234

Other player's hand:
W5 B5 B3 R1 W4
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 4 | 0.033 |  0.446  |    R     |
| Reveal 4 to P1 | 0.947 |  0.269  |    M     |
| Reveal 5 to P1 | 0.020 |  0.285  |          |

Value: 0.920

---

# Game State 36

## Game State

```
Information Tokens: 0.0
Life Tokens: 3.0
Fireworks: R4 Y0 G4 W3 B0
Deck size: 21.0
Discards:
R4 Y2 G1 G4 W1 W2 B2 B3

Hands:
Your hand:
WX || W25
XX || RYGB235
XX || RYGB235
XX || RYGB1235
X4 || RYGWB4

Other player's hand:
Y5 W3 Y2 R1 W1
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 2 | 1.000 |  1.000  |    ★     |

Value: 0.920

---

# Game State 37

## Game State

```
Information Tokens: 1.0
Life Tokens: 3.0
Fireworks: R4 Y0 G4 W3 B0
Deck size: 20.0
Discards:
R4 Y2 G1 G4 W1 W2 B2 B3 B5

Hands:
Your hand:
Y5 || Y5
XX || GWB234
YX || Y1234
XX || RGWB1234
XX || RYGWB1234

Other player's hand:
W5 B3 R1 W4 Y3
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 4 | 0.008 |  0.329  |    R     |
| Reveal W to P1 | 0.985 |  0.219  |    M     |
| Reveal 3 to P1 | 0.005 |  0.245  |          |
| Reveal 5 to P1 | 0.003 |  0.207  |          |

Value: 0.920

---

# Game State 38

## Game State

```
Information Tokens: 0.0
Life Tokens: 3.0
Fireworks: R4 Y0 G4 W3 B0
Deck size: 20.0
Discards:
R4 Y2 G1 G4 W1 W2 B2 B3 B5

Hands:
Your hand:
WX || W25
XX || RYGB235
XX || RYGB1235
W4 || W4
XX || RYGB12345

Other player's hand:
Y5 W3 Y2 R1 W1
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 2 | 0.003 |  0.501  |    R     |
|  PLAY card 4   | 0.997 |  0.499  |    M     |

Value: 0.920

---

# Game State 39

## Game State

```
Information Tokens: 0.0
Life Tokens: 3.0
Fireworks: R4 Y0 G4 W4 B0
Deck size: 19.0
Discards:
R4 Y2 G1 G4 W1 W2 B2 B3 B5

Hands:
Your hand:
Y5 || Y5
XX || GWB234
YX || Y1234
XX || RGWB1234
XX || RYGWB1234

Other player's hand:
W5 B3 R1 Y3 G1
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 2 | 1.000 |  1.000  |    ★     |

Value: 0.920

---

# Game State 40

## Game State

```
Information Tokens: 1.0
Life Tokens: 3.0
Fireworks: R4 Y0 G4 W4 B0
Deck size: 18.0
Discards:
R4 Y2 G1 G4 W1 W2 W3 B2 B3 B5

Hands:
Your hand:
WX || W25
XX || RYGB235
XX || RYGB1235
XX || RYGB12345
XX || RYGWB12345

Other player's hand:
Y5 Y2 R1 W1 B1
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 2 | 0.003 |  0.230  |          |
|  PLAY card 1   | 0.992 |  0.239  |    M     |
| Reveal 1 to P1 | 0.003 |  0.249  |          |
| Reveal 2 to P1 | 0.003 |  0.282  |    R     |

Value: 0.920

---

# Game State 41

## Game State

```
Information Tokens: 2.0
Life Tokens: 3.0
Fireworks: R4 Y0 G4 W5 B0
Deck size: 17.0
Discards:
R4 Y2 G1 G4 W1 W2 W3 B2 B3 B5

Hands:
Your hand:
Y5 || Y5
YX || Y1234
XX || RGWB1234
XX || RYGWB1234
XX || RYGWB12345

Other player's hand:
B3 R1 Y3 G1 R5
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 3 | 0.008 |  0.356  |          |
| Reveal 3 to P1 | 0.010 |  0.389  |    R     |
| Reveal 5 to P1 | 0.982 |  0.254  |    M     |

Value: 0.920

---

# Game State 42

## Game State

```
Information Tokens: 1.0
Life Tokens: 3.0
Fireworks: R4 Y0 G4 W5 B0
Deck size: 17.0
Discards:
R4 Y2 G1 G4 W1 W2 W3 B2 B3 B5

Hands:
Your hand:
XX || RYGB23
XX || RYGB123
XX || RYGB1234
XX || RYGWB1234
X5 || RYGWB5

Other player's hand:
Y5 Y2 R1 W1 B1
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 1 | 0.003 |  0.243  |    R     |
|  PLAY card 5   | 0.990 |  0.162  |    M     |
| Reveal W to P1 | 0.003 |  0.124  |          |
| Reveal 1 to P1 | 0.003 |  0.239  |          |
| Reveal 2 to P1 | 0.003 |  0.232  |          |

Value: 0.920

---

# Game State 43

## Game State

```
Information Tokens: 2.0
Life Tokens: 3.0
Fireworks: R5 Y0 G4 W5 B0
Deck size: 16.0
Discards:
R4 Y2 G1 G4 W1 W2 W3 B2 B3 B5

Hands:
Your hand:
Y5 || Y5
YX || Y1234
XX || RGWB1234
XX || RYGWB1234
XX || RYGWB12345

Other player's hand:
B3 R1 Y3 G1 Y3
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 3 | 0.799 |  0.325  |    M     |
| Reveal R to P1 | 0.163 |  0.331  |          |
| Reveal 3 to P1 | 0.038 |  0.344  |    R     |

Value: 0.920

---

# Game State 44

## Game State

```
Information Tokens: 3.0
Life Tokens: 3.0
Fireworks: R5 Y0 G4 W5 B0
Deck size: 15.0
Discards:
R1 R4 Y2 G1 G4 W1 W2 W3 B2 B3 B5

Hands:
Your hand:
XX || RYGB23
XX || RYGB123
XX || RYGB1234
XX || RYGWB1234
XX || RYGWB12345

Other player's hand:
Y5 Y2 W1 B1 B4
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 1 | 0.058 |  0.276  |          |
| Reveal W to P1 | 0.820 |  0.142  |    M     |
| Reveal 1 to P1 | 0.058 |  0.274  |          |
| Reveal 2 to P1 | 0.065 |  0.309  |    R     |

Value: 0.920

---

# Game State 45

## Game State

```
Information Tokens: 2.0
Life Tokens: 3.0
Fireworks: R5 Y0 G4 W5 B0
Deck size: 15.0
Discards:
R1 R4 Y2 G1 G4 W1 W2 W3 B2 B3 B5

Hands:
Your hand:
Y5 || Y5
YX || Y1234
WX || W1234
XX || RYGB12345
XX || RYGB12345

Other player's hand:
B3 R1 Y3 G1 Y3
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 3 | 0.028 |  0.336  |    R     |
| Reveal R to P1 | 0.028 |  0.335  |          |
| Reveal 3 to P1 | 0.945 |  0.329  |    M     |

Value: 0.920

---

# Game State 46

## Game State

```
Information Tokens: 1.0
Life Tokens: 3.0
Fireworks: R5 Y0 G4 W5 B0
Deck size: 15.0
Discards:
R1 R4 Y2 G1 G4 W1 W2 W3 B2 B3 B5

Hands:
Your hand:
X3 || RYGB3
XX || RYGB12
X3 || RYGB3
XX || RYGWB124
X3 || RYGWB3

Other player's hand:
Y5 Y2 W1 B1 B4
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 1 | 0.008 |  0.370  |    R     |
| Reveal 1 to P1 | 0.987 |  0.304  |    M     |
| Reveal 2 to P1 | 0.005 |  0.326  |          |

Value: 0.920

---

# Game State 47

## Game State

```
Information Tokens: 0.0
Life Tokens: 3.0
Fireworks: R5 Y0 G4 W5 B0
Deck size: 15.0
Discards:
R1 R4 Y2 G1 G4 W1 W2 W3 B2 B3 B5

Hands:
Your hand:
Y5 || Y5
YX || Y234
W1 || W1
X1 || RYGB1
XX || RYGB2345

Other player's hand:
B3 R1 Y3 G1 Y3
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 3 | 0.005 |  0.451  |          |
|  PLAY card 4   | 0.995 |  0.549  |    ★     |

Value: 0.920

---

# Game State 48

## Game State

```
Information Tokens: 0.0
Life Tokens: 3.0
Fireworks: R5 Y0 G4 W5 B1
Deck size: 14.0
Discards:
R1 R4 Y2 G1 G4 W1 W2 W3 B2 B3 B5

Hands:
Your hand:
X3 || RYGB3
XX || RYGB12
X3 || RYGB3
XX || RYGWB124
X3 || RYGWB3

Other player's hand:
Y5 Y2 W1 B4 R2
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 2 | 1.000 |  1.000  |    ★     |

Value: 0.920

---

# Game State 49

## Game State

```
Information Tokens: 1.0
Life Tokens: 3.0
Fireworks: R5 Y0 G4 W5 B1
Deck size: 13.0
Discards:
R1 R1 R4 Y2 G1 G4 W1 W2 W3 B2 B3 B5

Hands:
Your hand:
Y5 || Y5
YX || Y234
W1 || W1
XX || RYGB2345
XX || RYGWB12345

Other player's hand:
B3 Y3 G1 Y3 Y1
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 3 | 0.010 |  0.220  |          |
| Reveal Y to P1 | 0.010 |  0.251  |          |
| Reveal B to P1 | 0.008 |  0.207  |          |
| Reveal 1 to P1 | 0.972 |  0.322  |    ★     |

Value: 0.920

---

# Game State 50

## Game State

```
Information Tokens: 0.0
Life Tokens: 3.0
Fireworks: R5 Y0 G4 W5 B1
Deck size: 13.0
Discards:
R1 R1 R4 Y2 G1 G4 W1 W2 W3 B2 B3 B5

Hands:
Your hand:
X3 || RYGB3
X3 || RYGB3
X1 || RYGWB1
X3 || RYGWB3
X1 || RYGWB1

Other player's hand:
Y5 Y2 W1 B4 R2
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 3 | 1.000 |  1.000  |    ★     |

Value: 0.920

---

# Game State 51

## Game State

```
Information Tokens: 1.0
Life Tokens: 3.0
Fireworks: R5 Y0 G4 W5 B1
Deck size: 12.0
Discards:
R1 R1 R4 Y2 G1 G1 G4 W1 W2 W3 B2 B3 B5

Hands:
Your hand:
Y5 || Y5
YX || Y234
W1 || W1
XX || RYGB2345
XX || RYGWB12345

Other player's hand:
B3 Y3 Y3 Y1 B1
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 3 | 0.005 |  0.328  |          |
| Reveal Y to P1 | 0.005 |  0.363  |    R     |
| Reveal B to P1 | 0.990 |  0.310  |    M     |

Value: 0.920

---

# Game State 52

## Game State

```
Information Tokens: 0.0
Life Tokens: 3.0
Fireworks: R5 Y0 G4 W5 B1
Deck size: 12.0
Discards:
R1 R1 R4 Y2 G1 G1 G4 W1 W2 W3 B2 B3 B5

Hands:
Your hand:
B3 || B3
X3 || RYG3
X3 || RYGW3
X1 || RYGW1
BX || B12345

Other player's hand:
Y5 Y2 W1 B4 R2
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 2 | 0.003 |  0.441  |          |
|  PLAY card 4   | 0.997 |  0.559  |    ★     |

Value: 0.920

---

# Game State 53

## Game State

```
Information Tokens: 0.0
Life Tokens: 3.0
Fireworks: R5 Y1 G4 W5 B1
Deck size: 11.0
Discards:
R1 R1 R4 Y2 G1 G1 G4 W1 W2 W3 B2 B3 B5

Hands:
Your hand:
Y5 || Y5
YX || Y234
W1 || W1
XX || RYGB2345
XX || RYGWB12345

Other player's hand:
B3 Y3 Y3 B1 G5
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 3 | 1.000 |  1.000  |    ★     |

Value: 0.920

---

# Game State 54

## Game State

```
Information Tokens: 1.0
Life Tokens: 3.0
Fireworks: R5 Y1 G4 W5 B1
Deck size: 10.0
Discards:
R1 R1 R4 Y2 G1 G1 G4 W1 W1 W2 W3 B2 B3 B5

Hands:
Your hand:
B3 || B3
X3 || RYG3
X3 || RYGW3
BX || B12345
XX || RYGWB12345

Other player's hand:
Y5 Y2 B4 R2 G3
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 5 | 0.003 |  0.324  |          |
| Reveal R to P1 | 0.003 |  0.289  |          |
| Reveal 2 to P1 | 0.995 |  0.387  |    ★     |

Value: 0.920

---

# Game State 55

## Game State

```
Information Tokens: 0.0
Life Tokens: 3.0
Fireworks: R5 Y1 G4 W5 B1
Deck size: 10.0
Discards:
R1 R1 R4 Y2 G1 G1 G4 W1 W1 W2 W3 B2 B3 B5

Hands:
Your hand:
Y5 || Y5
Y2 || Y2
XX || RYGB345
X2 || RYGWB2
XX || RYGWB1345

Other player's hand:
B3 Y3 Y3 B1 G5
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 5 | 0.003 |  0.465  |          |
|  PLAY card 2   | 0.997 |  0.535  |    ★     |

Value: 0.920

---

# Game State 56

## Game State

```
Information Tokens: 0.0
Life Tokens: 3.0
Fireworks: R5 Y2 G4 W5 B1
Deck size: 9.0
Discards:
R1 R1 R4 Y2 G1 G1 G4 W1 W1 W2 W3 B2 B3 B5

Hands:
Your hand:
B3 || B3
X3 || RYG3
X3 || RYGW3
BX || B12345
XX || RYGWB12345

Other player's hand:
Y5 B4 R2 G3 B2
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 4 | 1.000 |  1.000  |    ★     |

Value: 0.920

---

# Game State 57

## Game State

```
Information Tokens: 1.0
Life Tokens: 3.0
Fireworks: R5 Y2 G4 W5 B1
Deck size: 8.0
Discards:
R1 R1 R4 Y2 G1 G1 G4 W1 W1 W2 W3 B1 B2 B3 B5

Hands:
Your hand:
Y5 || Y5
XX || RYGB345
X2 || RYGWB2
XX || RYGWB1345
XX || RYGWB12345

Other player's hand:
B3 Y3 Y3 G5 Y4
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 3 | 0.005 |  0.474  |          |
| Reveal Y to P1 | 0.995 |  0.526  |    ★     |

Value: 0.920

---

# Game State 58

## Game State

```
Information Tokens: 0.0
Life Tokens: 3.0
Fireworks: R5 Y2 G4 W5 B1
Deck size: 8.0
Discards:
R1 R1 R4 Y2 G1 G1 G4 W1 W1 W2 W3 B1 B2 B3 B5

Hands:
Your hand:
B3 || B3
Y3 || Y3
Y3 || Y3
XX || RGWB12345
YX || Y12345

Other player's hand:
Y5 B4 R2 G3 B2
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 4 | 0.008 |  0.568  |    R     |
|  PLAY card 2   | 0.992 |  0.432  |    M     |

Value: 0.920

---

# Game State 59

## Game State

```
Information Tokens: 0.0
Life Tokens: 3.0
Fireworks: R5 Y3 G4 W5 B1
Deck size: 7.0
Discards:
R1 R1 R4 Y2 G1 G1 G4 W1 W1 W2 W3 B1 B2 B3 B5

Hands:
Your hand:
Y5 || Y5
XX || RYGB345
X2 || RYGWB2
XX || RYGWB1345
XX || RYGWB12345

Other player's hand:
B3 Y3 G5 Y4 Y4
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 3 | 1.000 |  1.000  |    ★     |

Value: 0.920

---

# Game State 60

## Game State

```
Information Tokens: 1.0
Life Tokens: 3.0
Fireworks: R5 Y3 G4 W5 B1
Deck size: 6.0
Discards:
R1 R1 R2 R4 Y2 G1 G1 G4 W1 W1 W2 W3 B1 B2 B3 B5

Hands:
Your hand:
B3 || B3
Y3 || Y3
XX || RGWB12345
YX || Y12345
XX || RYGWB12345

Other player's hand:
Y5 B4 G3 B2 B1
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 2 | 0.040 |  0.309  |    R     |
| Reveal B to P1 | 0.887 |  0.155  |    M     |
| Reveal 1 to P1 | 0.048 |  0.259  |          |
| Reveal 2 to P1 | 0.025 |  0.277  |          |

Value: 0.920

---

# Game State 61

## Game State

```
Information Tokens: 0.0
Life Tokens: 3.0
Fireworks: R5 Y3 G4 W5 B1
Deck size: 6.0
Discards:
R1 R1 R2 R4 Y2 G1 G1 G4 W1 W1 W2 W3 B1 B2 B3 B5

Hands:
Your hand:
Y5 || Y5
BX || B345
XX || RYGW1345
BX || B12345
BX || B12345

Other player's hand:
B3 Y3 G5 Y4 Y4
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 3 | 1.000 |  1.000  |    ★     |

Value: 0.920

---

# Game State 62

## Game State

```
Information Tokens: 1.0
Life Tokens: 3.0
Fireworks: R5 Y3 G4 W5 B1
Deck size: 5.0
Discards:
R1 R1 R2 R4 Y2 G1 G1 G3 G4 W1 W1 W2 W3 B1 B2 B3 B5

Hands:
Your hand:
B3 || B3
Y3 || Y3
XX || RGWB12345
YX || Y12345
XX || RYGWB12345

Other player's hand:
Y5 B4 B2 B1 Y1
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 2 | 0.003 |  0.215  |          |
|  PLAY card 4   | 0.990 |  0.239  |    ★     |
| Reveal 1 to P1 | 0.003 |  0.183  |          |
| Reveal 2 to P1 | 0.003 |  0.222  |          |
| Reveal 4 to P1 | 0.003 |  0.141  |          |

Value: 0.920

---

# Game State 63

## Game State

```
Information Tokens: 1.0
Life Tokens: 3.0
Fireworks: R5 Y4 G4 W5 B1
Deck size: 4.0
Discards:
R1 R1 R2 R4 Y2 G1 G1 G3 G4 W1 W1 W2 W3 B1 B2 B3 B5

Hands:
Your hand:
Y5 || Y5
BX || B345
BX || B12345
BX || B12345
XX || RYGWB12345

Other player's hand:
B3 Y3 G5 Y4 W4
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 5 | 0.003 |  0.226  |          |
|  PLAY card 1   | 0.990 |  0.263  |    ★     |
| Reveal Y to P1 | 0.003 |  0.235  |          |
| Reveal W to P1 | 0.003 |  0.150  |          |
| Reveal 5 to P1 | 0.003 |  0.125  |          |

Value: 0.920

---

# Game State 64

## Game State

```
Information Tokens: 2.0
Life Tokens: 3.0
Fireworks: R5 Y5 G4 W5 B1
Deck size: 3.0
Discards:
R1 R1 R2 R4 Y2 G1 G1 G3 G4 W1 W1 W2 W3 B1 B2 B3 B5

Hands:
Your hand:
B3 || B3
Y3 || Y3
XX || RGWB12345
XX || RYGWB12345
XX || RYGWB12345

Other player's hand:
B4 B2 B1 Y1 G2
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 2 | 0.005 |  0.289  |          |
| Reveal 1 to P1 | 0.005 |  0.252  |          |
| Reveal 2 to P1 | 0.987 |  0.302  |    ★     |
| Reveal 4 to P1 | 0.003 |  0.156  |          |

Value: 0.920

---

# Game State 65

## Game State

```
Information Tokens: 1.0
Life Tokens: 3.0
Fireworks: R5 Y5 G4 W5 B1
Deck size: 3.0
Discards:
R1 R1 R2 R4 Y2 G1 G1 G3 G4 W1 W1 W2 W3 B1 B2 B3 B5

Hands:
Your hand:
BX || B345
B2 || B2
BX || B1345
XX || RYGWB1345
X2 || RYGWB2

Other player's hand:
B3 Y3 G5 Y4 W4
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 4 | 0.003 |  0.319  |    R     |
|  PLAY card 2   | 0.992 |  0.243  |    M     |
| Reveal Y to P1 | 0.003 |  0.274  |          |
| Reveal 5 to P1 | 0.003 |  0.164  |          |

Value: 0.920

---

# Game State 66

## Game State

```
Information Tokens: 1.0
Life Tokens: 3.0
Fireworks: R5 Y5 G4 W5 B2
Deck size: 2.0
Discards:
R1 R1 R2 R4 Y2 G1 G1 G3 G4 W1 W1 W2 W3 B1 B2 B3 B5

Hands:
Your hand:
B3 || B3
Y3 || Y3
XX || RGWB12345
XX || RYGWB12345
XX || RYGWB12345

Other player's hand:
B4 B1 Y1 G2 B4
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 2 | 0.003 |  0.251  |    R     |
|  PLAY card 1   | 0.990 |  0.244  |    M     |
| Reveal B to P1 | 0.003 |  0.150  |          |
| Reveal 1 to P1 | 0.003 |  0.215  |          |
| Reveal 4 to P1 | 0.003 |  0.139  |          |

Value: 0.920

---

# Game State 67

## Game State

```
Information Tokens: 1.0
Life Tokens: 3.0
Fireworks: R5 Y5 G4 W5 B3
Deck size: 1.0
Discards:
R1 R1 R2 R4 Y2 G1 G1 G3 G4 W1 W1 W2 W3 B1 B2 B3 B5

Hands:
Your hand:
BX || B345
BX || B1345
XX || RYGWB1345
X2 || RYGWB2
XX || RYGWB12345

Other player's hand:
Y3 G5 Y4 W4 Y1
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 4 | 0.003 |  0.261  |          |
|  PLAY card 1   | 0.992 |  0.331  |    ★     |
| Reveal Y to P1 | 0.003 |  0.252  |          |
| Reveal 5 to P1 | 0.003 |  0.156  |          |

Value: 0.920

---

# Game State 68

## Game State

```
Information Tokens: 2.0
Life Tokens: 3.0
Fireworks: R5 Y5 G5 W5 B4
Deck size: 0.0
Discards:
R1 R1 R2 R4 Y2 G1 G1 G3 G4 W1 W1 W2 W3 B1 B2 B3 B5

Hands:
Your hand:
BX || B1345
XX || RYGWB1345
X2 || RYGWB2
XX || RYGWB12345
XX || RYGWB12345

Other player's hand:
Y3 Y4 W4 Y1 XX (missing cards)
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 3 | 0.441 |  0.442  |          |
| Reveal Y to P1 | 0.559 |  0.558  |    ★     |

Value: 0.920

---

# Game State 69

## Game State

```
Information Tokens: 1.0
Life Tokens: 3.0
Fireworks: R5 Y5 G4 W5 B4
Deck size: 0.0
Discards:
R1 R1 R2 R4 Y2 G1 G1 G3 G4 W1 W1 W2 W3 B1 B2 B3 B5

Hands:
Your hand:
Y3 || Y3
XX || RGWB12345
XX || RYGWB12345
XX || RYGWB12345
XX || RYGWB12345

Other player's hand:
B1 Y1 G2 B4 R3
```

## Available Actions

|     Action     | MCTS  | Network | Selected |
| :------------: | :---: | :-----: | :------: |
| DISCARD card 1 | 0.013 |  0.380  |    R     |
|  PLAY card 2   | 0.975 |  0.241  |    M     |
| Reveal 1 to P1 | 0.013 |  0.379  |          |

Value: 0.920

---
