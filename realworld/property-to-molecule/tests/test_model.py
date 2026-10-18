"""Mechanism checks for the published CCOND architecture on a small vocabulary."""
import torch

from sedge_molecule.model import _GeneratorBase, ModelConfig
from sedge_molecule.losses import total_loss
from sedge_molecule.upstream import (TargetData, SpanningTreeVocabulary, smiles_list2atoms_list,
                                   get_max_valence_from_dataset, PAD_TOKEN)
from sedge_molecule.model import ConditionedCGenerator


def vocabulary():
    smiles = ["CCO", "CCN", "CCCO", "CCCCO"]
    tokens, valences = smiles_list2atoms_list(smiles, TOKEN2ATOMFEAT={}, VALENCES={})
    valences = get_max_valence_from_dataset(smiles, VALENCES=valences)
    return SpanningTreeVocabulary(tokens, valences, sort=True)


def test_zero_conditioner_preserves_base_parameters_and_rng():
    torch.set_num_threads(2)
    vocab = vocabulary()
    torch.manual_seed(17)
    base = _GeneratorBase(vocab, ModelConfig()).eval()
    rng_after_base = torch.get_rng_state().clone()
    torch.manual_seed(17)
    model = ConditionedCGenerator(vocab).eval()
    assert torch.equal(torch.get_rng_state(), rng_after_base)
    for key, value in base.state_dict().items():
        assert torch.equal(value, model.state_dict()[key]), key
    batch = TargetData.collate([TargetData.from_smiles("CCO", vocab, randomize_order=False).featurize()])
    z, observed = torch.tensor([[0.3, -0.2]]), torch.ones((1, 2), dtype=torch.bool)
    with torch.no_grad():
        conditioned = model(batch, z, observed, return_states=True)
        assert torch.count_nonzero(conditioned["states"]["C_delta"]) == 0
        assert torch.equal(conditioned["states"]["C_base"], conditioned["states"]["C_cond"])
        assert torch.equal(model.generation_logits(batch, z, observed), conditioned["logits"][:, -1].float())


def test_conditioner_receives_token_gradient_auxiliary_heads_ignore_targets():
    torch.set_num_threads(2)
    vocab = vocabulary()
    torch.manual_seed(17)
    model = ConditionedCGenerator(vocab).train()
    batch = TargetData.collate([TargetData.from_smiles(s, vocab, randomize_order=False).featurize() for s in ["CCO", "CCN"]])
    z = torch.tensor([[0.3, -0.2], [-0.5, 0.6]], requires_grad=True)
    observed = torch.ones_like(z, dtype=torch.bool)
    output = model(batch, z, observed)
    # The auxiliary outputs have no computational dependence on the supplied targets.
    gradient = torch.autograd.grad(output["property_mu"].sum() + output["property_sigma"].sum(),
                                   z, allow_unused=True, retain_graph=True)[0]
    assert gradient is None or torch.count_nonzero(gradient) == 0
    loss, _ = total_loss(output, batch[0], z, vocab.get_id(PAD_TOKEN))
    loss.backward()
    grad = model.c_condition.update[-1].weight.grad
    assert torch.isfinite(grad).all() and torch.count_nonzero(grad) > 0
    # Give the new path a nonzero update, then verify its target-dependent behavior.
    with torch.no_grad():
        model.c_condition.update[-1].weight.add_(grad, alpha=-1e-3)
    model.eval()
    with torch.no_grad():
        first = model(batch, z, observed, return_states=True)
        second = model(batch, z + 0.8, observed, return_states=True)
    assert torch.equal(first["states"]["C_base"], second["states"]["C_base"])
    assert torch.equal(first["property_mu"], second["property_mu"])
    assert not torch.equal(first["states"]["C_cond"], second["states"]["C_cond"])
