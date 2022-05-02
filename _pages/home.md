---
layout: splash
author_profile: true
---


<h1 align="center">Ler's do some Deep learning</h1>

<div class="cards">
  <div class="card">BERT RL</div>
  <div class="card">City Skylines</div>
  <div class="card">Option Pricing</div>
  <div class="card">Fractals Crack</div>
  <div class="card">Blue Locator</div>
  <div class="card">Sixers</div>
</div>

<style>

.card {
  background-color: black;
  color: white;
  padding: 1rem;
  height: 4rem;
}

.cards {
  max-width: 1200px;
  margin: 50px auto;
  display: grid;
  grid-gap: 1rem;
}

/* Screen larger than 600px? 2 column */
@media (min-width: 600px) {
  .cards { grid-template-columns: repeat(2, 1fr); }
}

/* Screen larger than 900px? 3 columns */
@media (min-width: 900px) {
  .cards { grid-template-columns: repeat(3, 1fr); }
}
</style>