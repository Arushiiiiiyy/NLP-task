# ga/evolve.py
population = initial_gemini_texts()

for gen in range(10):
    scores = [(p, human_prob(p)) for p in population]
    elites = sorted(scores, key=lambda x: x[1], reverse=True)[:3]

    population = []
    for e,_ in elites:
        mutated = generate(
            f"Rewrite this paragraph to alter rhythm subtly:\n{e}"
        )
        population.append(mutated)
