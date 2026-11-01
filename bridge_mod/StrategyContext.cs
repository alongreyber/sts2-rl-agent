// Persistent inventory is attached on the game thread, before awaiting a choice.
using System.Linq;
using System.Text.Json.Nodes;
using MegaCrit.Sts2.Core.Context;
using MegaCrit.Sts2.Core.Runs;

namespace STS2BridgeMod;

public static class StrategyContext
{
    public static string[] Routes(MegaCrit.Sts2.Core.Map.MapPoint point, int depth = 3)
    {
        if (depth == 0 || !point.Children.Any()) return new[] { point.PointType.ToString() };
        return point.Children.SelectMany(c => Routes(c, depth - 1)
            .Select(tail => point.PointType.ToString() + ">" + tail)).ToArray();
    }
    public static string Attach(string json)
    {
        var state = RunManager.Instance.DebugOnlyGetState();
        var player = LocalContext.GetMe(state);
        if (player == null) return json;
        var root = JsonNode.Parse(json)!.AsObject();
        root["strategy_context"] = System.Text.Json.JsonSerializer.SerializeToNode(new
        {
            act = state.CurrentActIndex + 1,
            floor = state.TotalFloor,
            hp = player.Creature.CurrentHp,
            max_hp = player.Creature.MaxHp,
            gold = player.Gold,
            deck = player.Deck.Cards.Select(c => new { id = c.Id.Entry, upgraded = c.IsUpgraded }).ToArray(),
            relics = player.Relics.Select(r => r.Id.Entry).ToArray(),
            potions = player.Potions.Select(p => p.Id.Entry).ToArray(),
        });
        return root.ToJsonString();
    }
}
