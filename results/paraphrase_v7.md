# Retrieval under paraphrase (batch v7 follow-up)

Hit = the retriever picks the needle turn among the earlier user turns at the final question (all k = 2, 4, 8; 180 dialogues per set). Templates fixed before results.

## words

| question | example | lexical | embed |
|---|---|---|---|
| original | What is my colleague's rabbit called, and in which city does it live? Answer with the name and the city only. | 180 / 180 | 180 / 180 |
| P1 name/where | What's the name of my colleague's rabbit, and where does it live? | 180 / 180 | 180 / 180 |
| P2 pet/town | Do you remember what my colleague's pet is named and which town it's in? | 0 / 180 | 180 / 180 |
| P3 belongs to | Remind me: the rabbit that belongs to my colleague - what's it called and where? | 180 / 180 | 180 / 180 |
| P4 no entity | Earlier I mentioned a pet. What was its name and home city? | 0 / 180 | 177 / 180 |
| P5 has a | My colleague has a rabbit; tell me its name and the city it lives in. | 180 / 180 | 180 / 180 |

## digits

| question | example | lexical | embed |
|---|---|---|---|
| original | What is the access code for the silent harbor C6? Answer with the number only. | 180 / 180 | 180 / 180 |
| P1 code for | What's the code for the silent harbor C6? | 180 / 180 | 180 / 180 |
| P2 get into | I need to get into the silent harbor. What were the digits? | 176 / 180 | 180 / 180 |
| P3 noun only | Which number did I give you to open the harbor? | 0 / 180 | 180 / 180 |
| P4 no entity | What was the access code I told you earlier? | 180 / 180 | 180 / 180 |
| P5 PIN | Can you recall the PIN I mentioned? | 0 / 180 | 179 / 180 |

