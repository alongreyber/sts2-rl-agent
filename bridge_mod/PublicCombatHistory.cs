using System;
using System.Collections.Generic;
using System.Linq;
using MegaCrit.Sts2.Core.Combat;
using MegaCrit.Sts2.Core.Combat.History.Entries;
using MegaCrit.Sts2.Core.Entities.Cards;
using MegaCrit.Sts2.Core.Entities.Players;

namespace STS2BridgeMod;

internal static class PublicCombatHistory
{
    internal static Dictionary<string, int> Of(Player player, ICombatState state)
    {
        var entries = CombatManager.Instance.History.Entries.Where(e => e.Actor == player.Creature).ToArray();
        var result = new Dictionary<string, int>();
        foreach (var suffix in new[] { "turn", "combat" })
        {
            var plays = entries.OfType<CardPlayFinishedEntry>().Where(e => suffix == "combat" || e.HappenedThisTurn(state)).ToArray();
            result["plays_" + suffix] = plays.Length;
            result["attacks_" + suffix] = plays.Count(e => e.CardPlay.Card.Type == CardType.Attack);
            result["skills_" + suffix] = plays.Count(e => e.CardPlay.Card.Type == CardType.Skill);
            result["powers_" + suffix] = plays.Count(e => e.CardPlay.Card.Type == CardType.Power);
        }
        var turn = entries.Where(e => e.HappenedThisTurn(state)).ToArray();
        result["draws_turn"] = turn.OfType<CardDrawnEntry>().Count();
        result["discards_turn"] = turn.OfType<CardDiscardedEntry>().Count();
        result["exhausts_turn"] = turn.OfType<CardExhaustedEntry>().Count();
        result["energy_spent_turn"] = turn.OfType<EnergySpentEntry>().Sum(e => e.Amount);
        result["plays_last_turn"] = entries.OfType<CardPlayFinishedEntry>().Count(e => e.HappenedLastPlayerTurn(player));
        return result;
    }
}
