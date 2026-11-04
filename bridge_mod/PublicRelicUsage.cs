using System.Collections.Generic;
using MegaCrit.Sts2.Core.Models;
namespace STS2BridgeMod;
internal static class PublicRelicUsage
{
    internal static Dictionary<string, bool> Of(RelicModel relic)
    {
        var result = new Dictionary<string, bool> { ["used_up"] = relic.IsUsedUp };
        foreach (var pair in new[] { ("UsedThisCombat", "used_this_combat"), ("UsedThisTurn", "used_this_turn"), ("TriggeredThisTurn", "triggered_this_turn") })
        {
            var property = relic.GetType().GetProperty(pair.Item1);
            if (property?.PropertyType == typeof(bool)) result[pair.Item2] = (bool)property.GetValue(relic)!;
        }
        return result;
    }
}
