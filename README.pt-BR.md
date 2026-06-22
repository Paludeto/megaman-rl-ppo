# Mega Man (NES) — Derrotando Bosses Utilizando PPO

*Leia em outro idioma: [English](README.md) · **Português***

Treinamento por reforço profundo (PPO, `CnnPolicy`, **sem LSTM**) que aprende a derrotar
chefes de *Mega Man* (NES) usando **apenas os pixels da tela** como observação. O mesmo
conjunto de hiperparâmetros e a mesma função de recompensa genérica vencem o **Yellow Devil**
e os **6 Robot Masters** (Bombman, Cutman, Elecman, Fireman, Gutsman, Iceman).

- **Observação:** quadro RGB → tons de cinza 84×84 → empilhamento de 4 quadros (`84×84×4`).
- **Ações:** `MULTI_DISCRETE` (direção, A, B).
- **Algoritmo:** PPO + GAE (stable-baselines3), seed fixa (666).
- **Recompensa genérica:** `r = d · (acertos − m · dano)`, com `d = 0,05`. Sem bônus de vitória
  na avaliação. A vitória é detectada pela RAM `boss_health` (endereço 1729, genérico entre chefes).

---

## Reprodução em 4 passos (TL;DR)

```bash
# 1. ambiente
conda env create -f environment.yml && conda activate megaman-ppo

# 2. ROM (você fornece — veja a seção 2)
cp /caminho/megaman.nes custom_integrations/MegaMan-v1-Nes/rom.nes

# 3. treinar um chefe (ex.: Bombman)
python src/train.py --tag bombman --state Bombman-boss --n-hits 14

# 4. avaliar o modelo treinado
python src/eval_win.py --model models/bombman_best/best_model.zip --state Bombman-boss --episodes 10
```

---

## 1. Requisitos

Use **conda**:

```bash
conda env create -f environment.yml   # cria o env "megaman-ppo" (Python 3.12) e instala tudo
conda activate megaman-ppo
```

O `environment.yml` instala as dependências reaproveitando o `requirements.txt`.
`stable-baselines3` puxa `torch`, `numpy`, `gymnasium` e `cloudpickle` compatíveis.
GPU NVIDIA é fortemente recomendada (o treino é viável em CPU, mas lento).

## 2. ROM (obrigatório — você fornece o seu)

> ⚠️ Por questões legais, **o ROM não é distribuído neste repositório.** Você precisa de uma
> cópia legalmente adquirida de *Mega Man (NES)*.

Coloque o arquivo como `rom.nes` dentro da pasta de integração e confira o hash:

```bash
cp /caminho/do/seu/megaman.nes custom_integrations/MegaMan-v1-Nes/rom.nes
sha1sum custom_integrations/MegaMan-v1-Nes/rom.nes
# deve ser: 2f88381557339a14c20428455f6991c1eb902c99
```

Os arquivos de integração (`data.json`, `scenario.json`, `metadata.json`, `rom.sha`) e os
*save states* de cada chefe (`*.state`) já estão incluídos — só falta o `rom.nes`.

## 3. Estrutura

```
src/
  wrappers.py       # frameskip, recompensa (BossWrapper), pré-processamento 84×84 (WarpFrame)
  env.py            # make_env / make_venv: registra a integração custom e vetoriza
  train.py          # treino PPO (CLI parametrizada por --tag)
  eval_win.py       # avalia um modelo salvo: taxa de vitória, recompensa e duração médias
custom_integrations/MegaMan-v1-Nes/   # integração stable-retro (sem o rom.nes)
environment.yml     # ambiente conda
requirements.txt    # dependências (reusado pelo environment.yml)
```

Saídas geradas em tempo de execução (ignoradas pelo git): `logs/<tag>`, `checkpoints/<tag>`,
`models/<tag>_best/best_model.zip`.

## 4. Treinar

A **receita é fixa** (hiperparâmetros do PPO + função de recompensa) e está nas constantes no
topo de `src/train.py`. A CLI expõe só o que muda por chefe — no caso geral, três flags:

```bash
python src/train.py --tag <tag> --state <STATE> --n-hits <N>
```

O **early-stop** encerra assim que a recompensa de avaliação cruza o limiar de vitória
`(n_hits − 0,5)·d`. Parâmetros por chefe:

| Chefe        | `--state`            | `--n-hits` | Extra                                       |
|--------------|----------------------|:----------:|---------------------------------------------|
| Yellow Devil | `YellowDevil-boss`   | 14         | `--align-bonus 0.10 --episode-frames 14400` |
| Bombman      | `Bombman-boss`       | 14         | —                                           |
| Cutman       | `Cutman-boss`        | 10         | —                                           |
| Gutsman      | `Gutsman-boss`       | 14         | —                                           |
| Elecman      | `Elecman-boss`       | 28         | —                                           |
| Iceman       | `Iceman-boss`        | 28         | —                                           |
| Fireman      | `Fireman-boss`       | 14         | —                                           |

Yellow Devil é o único caso especial — precisa do *reward shaping* de mira (`--align-bonus`) e
de episódios mais longos:

```bash
python src/train.py --tag yd --state YellowDevil-boss --n-hits 14 \
  --align-bonus 0.10 --episode-frames 14400
```

Flags opcionais: `--timesteps` (teto, default 40 M), `--n-envs` (default 16), `--checkpoint`
(retomar), `--visualize`. Acompanhe com TensorBoard:

```bash
tensorboard --logdir logs/
```

### Reproduzir os 6 Robot Masters de uma vez

```bash
declare -A NHITS=( [Bombman-boss]=14 [Cutman-boss]=10 [Gutsman-boss]=14 \
                   [Elecman-boss]=28 [Iceman-boss]=28 [Fireman-boss]=14 )
for st in "${!NHITS[@]}"; do
  python src/train.py --tag "${st%-boss}" --state "$st" --n-hits "${NHITS[$st]}"
done
```

## 5. Avaliar

```bash
python src/eval_win.py --model models/<tag>_best/best_model.zip --state <STATE> --episodes 10
```

Carrega o modelo salvo, roda os episódios na tarefa real (sem *shaping*) e imprime a taxa de
vitória, a recompensa média e a duração média. Use `--deterministic` para a política gulosa.
Para o Yellow Devil acrescente `--episode-frames 14400` (igual ao treino).

## 6. Resultados esperados

Recompensa de avaliação flawless = `n_hits · d` (vitória sem sofrer dano). O *early-stop* só
dispara nesse teto, então 5 dos 7 chefes convergem para o flawless exato; Iceman e Fireman
vencem de forma **robusta** (consistente, mas sofrendo algum dano).

| Chefe        | Recompensa eval | Vitória  | Custo aprox. (passos até o platô) |
|--------------|:---------------:|----------|:---------------------------------:|
| Gutsman      | 0,70 (flawless) | 100%     | ~1,3 M                            |
| Iceman       | 1,30 (robusto)  | 100%     | ~1,9 M                            |
| Fireman      | 0,55 (robusto)  | 100%     | ~2,0 M                            |
| Bombman      | 0,70 (flawless) | 100%     | ~2,5 M                            |
| Cutman       | 0,50 (flawless) | 100%     | ~2,8 M                            |
| Elecman      | 1,40 (flawless) | 100%     | ~2,9 M                            |
| Yellow Devil | 0,70 (flawless) | 100%     | ~15,7 M                           |

Em GPU, os Robot Masters convergem em minutos a ~1 h cada; o Yellow Devil é o mais caro
(~15 M passos). A seed é fixa (666), mas há estocasticidade no emulador/política — os números
acima são o alvo, com pequenas variações entre execuções.

## 7. Estender para um novo chefe

`n_hits = ceil(28 / dano_por_tiro)` (HP do chefe = 28). Ex.: *buster* (2 de dano) → 14; uma arma
de 4 de dano → 7. Para treinar um chefe novo: gere seu `.state` inicial, descubra o dano por
tiro da arma usada, calcule o `n_hits` e use a receita base da seção 4.

## 8. Notas de design

- **Sem LSTM e sem currículo:** experimentos mostraram que ambos são dispensáveis. O que
  importa é a recompensa alinhada (no Yellow Devil, o `--align-bonus` aproxima a altura do
  tiro à do olho via *potential-based reward shaping*) e episódios longos o suficiente.
- **Normalização de recompensa** (fixa na receita) afeta apenas a recompensa, não a observação,
  então é segura para avaliação. A avaliação usa sempre a tarefa real, sem *shaping*.

## 9. Problemas comuns

- **Hash do ROM diferente** de `2f88381…`: você tem outra revisão/dump; a integração espera
  exatamente esse ROM. Sem o `rom.nes` correto, o `stable_retro.make` falha.
- **Avaliação não vence** um modelo que venceu no treino: confirme que o `--episode-frames` da
  avaliação é idêntico ao do treino (no Yellow Devil, `14400`).

## Licença

Veja [LICENSE](LICENSE). O ROM de *Mega Man* **não** está incluído e não é coberto por esta licença.
