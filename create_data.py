#!/usr/bin/env python3
"""
create_data.py — three options, no HuggingFace needed

Usage:
    python create_data.py                    # tries PTB download, falls back to builtin
    python create_data.py --source builtin   # large built-in corpus, always works
    python create_data.py --source local --file your_corpus.txt
"""
import argparse, re, json, collections, urllib.request

parser = argparse.ArgumentParser()
parser.add_argument('--source', default='auto', choices=['auto','ptb','local','builtin'])
parser.add_argument('--file',   default=None)
parser.add_argument('--vocab_size', type=int, default=1024)
args = parser.parse_args()

BUILTIN = """
the history of science includes many discoveries that changed human understanding of nature
mathematics provides language for describing physical laws and natural phenomena precisely
number theory studies integers properties prime factorization divisibility and congruences
fundamental theorem arithmetic states every integer greater one factors uniquely into primes
topology examines properties preserved continuous deformations without tearing or gluing spaces
group theory studies symmetries algebraic structure through abstract mathematical objects transformations
differential geometry extends calculus curved spaces manifolds rich geometric structure tensors
algebraic geometry studies geometric objects defined polynomial equations various fields varieties
category theory provides unified language describing mathematical structures morphisms and functors
physics describes fundamental laws governing matter energy space time universe conservation symmetry
classical mechanics motion macroscopic objects forces trajectories orbits lagrangian hamiltonian
quantum mechanics probabilistic description microscopic particles wave functions operators uncertainty
special relativity unifies space time spacetime speed light invariant inertial frames lorentz
general relativity gravity curvature spacetime mass energy stress tensor geodesics black holes
thermodynamics heat work temperature entropy free energy phase transitions equilibrium irreversibility
electromagnetism electric magnetic fields maxwell equations waves radiation photons polarization
statistical mechanics microscopic particle behavior macroscopic thermodynamic bulk properties partition
fluid dynamics motion liquids gases turbulence viscosity pressure velocity navier stokes equations
quantum field theory combines mechanics special relativity particles fields interactions propagators
condensed matter physics properties ground states excitations quantum phases superconductivity
nuclear physics atomic nuclei protons neutrons binding energy radioactive decay fission fusion
particle physics fundamental constituents matter quarks leptons bosons gauge symmetries standard model
astrophysics stars galaxies black holes neutron pulsars supernovae gravitational waves cosmology
computer science algorithms data structures computation information processing systems analysis design
algorithms procedures solving computational problems inputs outputs efficiency correctness complexity
data structures arrays lists trees graphs hash tables heaps balanced search efficient operations
machine learning patterns data predictions decisions gradient optimization regularization generalization
neural networks hierarchical representations composing layers parameterized nonlinear activation functions
gradient descent optimizes parameters iteratively steepest loss decrease learning rate momentum
backpropagation computes gradients computational graphs chain rule automatic differentiation efficient
attention mechanisms selectively focus relevant parts input context queries keys values weighted sum
transformers sequences self attention feed forward layers parallel residual connections normalization
language models predict probability token given preceding context autoregressive beam search sampling
convolutional networks local features images translation equivariance pooling stride filters kernels
recurrent networks sequences hidden state across time steps gating memory forgetting vanishing gradient
reinforcement learning optimal behavior trial error reward signals exploration exploitation policy value
computer vision detect segment classify recognize objects scenes depth estimation tracking video
natural language processing semantics syntax parsing translation summarization question answering
cryptography secure communication encryption decryption signatures public key asymmetric protocols
operating systems processes memory scheduling input output file systems virtualization concurrency
database systems store retrieve query transactions consistency isolation durability normalization indexes
distributed systems consensus replication fault tolerance availability consistency partition tolerance
biology living organisms structure function growth evolution ecology systematically classification
cell biology structural functional unit organelles membrane cytoskeleton signaling vesicles division
genetics traits inherited chromosomes dna rna expression regulation epigenetics mutation selection
evolution populations change generations natural selection fitness adaptation speciation phylogeny
neuroscience nervous system brain spinal cord neurons synapses plasticity circuits cognition behavior
molecular biology dna replication transcription translation ribosomes codons proteins folding structure
ecology food webs nutrient cycles population dynamics succession diversity stability keystone species
immunology pathogens antibodies lymphocytes inflammation complement vaccination immune memory response
developmental biology stem cells differentiation morphogenesis patterning induction tissue organ systems
biochemistry enzymes metabolism cofactors energy coupling glycolysis krebs cycle oxidative phosphorylation
chemistry composition structure properties transformations matter atomic molecular scales bonding
organic chemistry carbon compounds reactions mechanisms synthesis functional groups spectroscopy analysis
physical chemistry thermodynamics kinetics quantum mechanics applied chemical systems equilibria
inorganic chemistry metals coordination compounds main group transition elements periodic trends
analytical chemistry separating identifying quantifying chemical substances mixtures samples
polymer chemistry macromolecules monomers chain growth condensation properties applications materials
electrochemistry electron transfer redox reactions batteries fuel cells corrosion electroplating
economics individuals firms societies allocate scarce resources unlimited wants preferences constraints
microeconomics consumer firm behavior supply demand elasticity costs production profit markets
macroeconomics output employment inflation monetary policy fiscal multipliers business cycles growth
game theory strategic equilibrium cooperation competition auctions mechanism design incentives
behavioral economics cognitive biases heuristics prospect theory framing anchoring bounded rationality
financial markets asset pricing risk return portfolio optimization derivatives options futures
international trade comparative advantage gains exchange specialization protectionism tariffs quotas
monetary policy central banks interest rates inflation targeting quantitative easing reserve requirements
fiscal policy spending taxation automatic stabilizers deficits debt sustainability crowding multiplier
history civilizations cultures events causes consequences change continuity significance context
ancient mesopotamia egypt greece rome civilizations writing law philosophy science art literature
medieval europe feudalism christianity islamic ottoman byzantine mongol renaissance printing press
enlightenment scientific revolution reason progress secularism rights liberty democracy republics
industrial capitalism steam factories urbanization class labor unions socialism imperialism colonies
philosophy fundamental questions existence knowledge value reason mind language society politics
metaphysics reality consciousness free will determinism causation time identity mind body problem
epistemology knowledge justification belief truth rationalism empiricism skepticism coherence reliabilism
ethics consequentialism deontology virtue contractualism rights duties care justice fairness
political philosophy justice liberty equality authority legitimacy democracy sovereignty constitution
""" * 10

def tokenize(text):
    return re.findall(r'[a-z]+', text.lower())

# Get corpus
if args.source in ('auto', 'ptb'):
    try:
        print("Downloading PTB corpus...", flush=True)
        url = 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt'
        with urllib.request.urlopen(url, timeout=15) as r:
            train_text = r.read().decode('utf-8')
        url2 = 'https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt'
        with urllib.request.urlopen(url2, timeout=15) as r:
            val_text = r.read().decode('utf-8')
        print(f"  PTB downloaded: {len(train_text):,} train chars")
    except Exception as e:
        print(f"  Download failed ({e}), using builtin corpus")
        args.source = 'builtin'

if args.source in ('builtin',):
    n = int(0.9 * len(BUILTIN))
    train_text = BUILTIN[:n]; val_text = BUILTIN[n:]

elif args.source == 'local':
    assert args.file, "Need --file with --source local"
    with open(args.file) as f: text = f.read()
    n = int(0.9*len(text)); train_text=text[:n]; val_text=text[n:]

train_words = tokenize(train_text)
val_words   = tokenize(val_text)
freq = collections.Counter(train_words + val_words)
special = ['<pad>', '<unk>', '<eos>']
top = [w for w,_ in freq.most_common(args.vocab_size - len(special))]
vocab = special + top
w2i = {w:i for i,w in enumerate(vocab)}
train_ids = [w2i.get(w,1) for w in train_words]
val_ids   = [w2i.get(w,1) for w in val_words]

print(f"\n  Train: {len(train_ids):,} tokens")
print(f"  Val:   {len(val_ids):,} tokens")
print(f"  Vocab: {len(vocab)}")
print(f"  UNK:   {100*(train_ids.count(1)+val_ids.count(1))/(len(train_ids)+len(val_ids)):.1f}%")
print(f"  Sample: {[vocab[i] for i in train_ids[:6]]}")

with open('/tmp/train_ids.json','w') as f: json.dump(train_ids,f)
with open('/tmp/val_ids.json',  'w') as f: json.dump(val_ids,f)
with open('/tmp/vocab.json',    'w') as f: json.dump(vocab,f)
print("\nSaved. Now run: python monodromy_training.py")
