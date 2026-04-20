# MC-TextWorld Task Discovery

## Criteria

- `official_plan_replayable`: `tasks.json + plans.json` compile to known `action_lib.json` actions and replay successfully.
- `dependency_graph_buildable`: a quantity-aware executable action sequence can be synthesized from `action_lib.json` for the task goal.

## Summary

- Total tasks: `208`
- Unique task ids: `207`
- Total plans: `211`
- Official-plan replayable tasks: `96` unique ids
- Dependency-graph buildable tasks: `192` unique ids
- 67-goal tasks present in original tasks.json: `63` unique ids
- 67-goal official-plan replayable tasks: `57` unique ids
- 67-goal dependency-graph buildable tasks: `63` unique ids
- 67-goal synthetic dependency-graph buildable targets: `67` / `67`
- 67-goal targets missing from original tasks.json: `blast_furnace, iron_axe, stonecutter, torch`

## By Group

| Group | Tasks | Official Replayable | Graph Buildable | Goals-67 | Goals-67 Buildable |
| --- | ---: | ---: | ---: | ---: | ---: |
| animals | 3 | 3 | 3 | 0 | 0 |
| brewing | 2 | 2 | 2 | 0 | 0 |
| building | 18 | 10 | 18 | 0 | 0 |
| colorful | 47 | 0 | 47 | 0 | 0 |
| combat | 5 | 5 | 5 | 5 | 5 |
| decoration | 11 | 7 | 11 | 7 | 7 |
| decorations | 1 | 0 | 1 | 0 | 0 |
| equipment | 11 | 7 | 11 | 7 | 7 |
| foodstuffs | 7 | 7 | 7 | 1 | 1 |
| hand | 2 | 1 | 2 | 0 | 0 |
| iron | 6 | 6 | 6 | 5 | 5 |
| miscellaneous | 9 | 8 | 9 | 5 | 5 |
| occupation_block | 5 | 5 | 5 | 2 | 2 |
| redstone | 27 | 3 | 27 | 7 | 7 |
| test | 23 | 6 | 8 | 0 | 0 |
| tool | 25 | 24 | 25 | 23 | 23 |
| transportation | 6 | 2 | 6 | 1 | 1 |

## Dependency-Graph Buildable Tasks

- **animals** (3): white_bed, white_carpet, white_wool
- **brewing** (2): cauldron, glass_bottle
- **building** (18): acacia_log, acacia_planks, acacia_slab, acacia_wood, birch_log, birch_planks, birch_slab, birch_wood, glass, jungle_log, jungle_planks, jungle_slab, jungle_wood, oak_log, oak_planks, oak_slab, oak_wood, stone
- **colorful** (46): blue_banner, blue_bed, blue_carpet, blue_dye, blue_wool, light_blue_banner, light_blue_bed, light_blue_carpet, light_blue_dye, light_blue_wool, light_gray_banner, light_gray_bed, light_gray_carpet, light_gray_dye, light_gray_wool, magenta_banner, magenta_bed, magenta_carpet, magenta_dye, magenta_wool, orange_banner, orange_bed, orange_carpet, orange_dye, orange_wool, pink_banner, pink_bed, pink_carpet, pink_dye, pink_wool, purple_banner, purple_bed, purple_carpet, purple_dye, purple_wool, red_banner, red_bed, red_carpet, red_dye, red_wool, white_dye, yellow_banner, yellow_bed, yellow_carpet, yellow_dye, yellow_wool
- **combat** (5): diamond_sword, golden_sword, iron_sword, stone_sword, wooden_sword
- **decoration** (11): acacia_fence, birch_fence, chain, chest, crafting_table, furnace, iron_bars, jukebox, jungle_fence, ladder, oak_fence
- **decorations** (1): item_frame
- **equipment** (11): diamond_boots, diamond_chestplate, diamond_helmet, diamond_leggings, iron_boots, iron_helmet, leather_boots, leather_chestplate, leather_helmet, leather_leggings, shield
- **foodstuffs** (7): bowl, cooked_beef, cooked_chicken, cooked_mutton, cooked_porkchop, gold_nugget, golden_apple
- **hand** (2): painting, white_banner
- **iron** (6): heavy_weighted_pressure_plate, hopper, iron_chestplate, iron_leggings, iron_nugget, shears
- **miscellaneous** (9): book, bucket, charcoal, diamond, gold_ingot, iron_ingot, iron_ore, paper, stick
- **occupation_block** (5): barrel, composter, loom, smithing_table, smoker
- **redstone** (27): acacia_button, acacia_door, acacia_fence_gate, acacia_trapdoor, activator_rail, birch_button, birch_door, birch_fence_gate, birch_trapdoor, compass, dropper, iron_door, iron_trapdoor, jungle_button, jungle_door, jungle_fence_gate, jungle_trapdoor, note_block, oak_button, oak_door, oak_fence_gate, oak_trapdoor, piston, redstone, redstone_block, redstone_torch, tripwire_hook
- **test** (8): test_birch_log, test_diamond, test_gold, test_iron, test_oak_log, test_obsidian, test_redstone, test_sand
- **tool** (25): clock, crossbow, diamond_axe, diamond_hoe, diamond_pickaxe, diamond_shovel, golden_axe, golden_boots, golden_chestplate, golden_helmet, golden_hoe, golden_leggings, golden_pickaxe, golden_shovel, iron_hoe, iron_pickaxe, iron_shovel, stone_axe, stone_hoe, stone_pickaxe, stone_shovel, wooden_axe, wooden_hoe, wooden_pickaxe, wooden_shovel
- **transportation** (6): acacia_boat, birch_boat, jungle_boat, minecart, oak_boat, rail

## Official-Plan Replayable Tasks

- **animals** (3): white_bed, white_carpet, white_wool
- **brewing** (2): cauldron, glass_bottle
- **building** (10): acacia_log, acacia_wood, birch_log, birch_wood, glass, jungle_log, jungle_wood, oak_log, oak_wood, stone
- **combat** (5): diamond_sword, golden_sword, iron_sword, stone_sword, wooden_sword
- **decoration** (7): chain, chest, crafting_table, furnace, iron_bars, jukebox, ladder
- **equipment** (7): diamond_boots, diamond_chestplate, diamond_helmet, diamond_leggings, iron_boots, iron_helmet, shield
- **foodstuffs** (7): bowl, cooked_beef, cooked_chicken, cooked_mutton, cooked_porkchop, gold_nugget, golden_apple
- **hand** (1): white_banner
- **iron** (6): heavy_weighted_pressure_plate, hopper, iron_chestplate, iron_leggings, iron_nugget, shears
- **miscellaneous** (8): bucket, charcoal, diamond, gold_ingot, iron_ingot, iron_ore, paper, stick
- **occupation_block** (5): barrel, composter, loom, smithing_table, smoker
- **redstone** (3): iron_door, iron_trapdoor, tripwire_hook
- **test** (6): test_birch_log, test_diamond, test_gold, test_oak_log, test_obsidian, test_sand
- **tool** (24): crossbow, diamond_axe, diamond_hoe, diamond_pickaxe, diamond_shovel, golden_axe, golden_boots, golden_chestplate, golden_helmet, golden_hoe, golden_leggings, golden_pickaxe, golden_shovel, iron_hoe, iron_pickaxe, iron_shovel, stone_axe, stone_hoe, stone_pickaxe, stone_shovel, wooden_axe, wooden_hoe, wooden_pickaxe, wooden_shovel
- **transportation** (2): minecart, rail

## Goals-67 Dependency-Graph Buildable Tasks

- **combat** (5): diamond_sword, golden_sword, iron_sword, stone_sword, wooden_sword
- **decoration** (7): chain, chest, crafting_table, furnace, iron_bars, jukebox, ladder
- **equipment** (7): diamond_boots, diamond_chestplate, diamond_helmet, diamond_leggings, iron_boots, iron_helmet, shield
- **foodstuffs** (1): bowl
- **iron** (5): hopper, iron_chestplate, iron_leggings, iron_nugget, shears
- **miscellaneous** (5): bucket, charcoal, diamond, gold_ingot, stick
- **occupation_block** (2): smithing_table, smoker
- **redstone** (7): activator_rail, compass, dropper, note_block, piston, redstone_torch, tripwire_hook
- **tool** (23): diamond_axe, diamond_hoe, diamond_pickaxe, diamond_shovel, golden_axe, golden_boots, golden_chestplate, golden_helmet, golden_hoe, golden_leggings, golden_pickaxe, golden_shovel, iron_hoe, iron_pickaxe, iron_shovel, stone_axe, stone_hoe, stone_pickaxe, stone_shovel, wooden_axe, wooden_hoe, wooden_pickaxe, wooden_shovel
- **transportation** (1): rail

## Exclusions

Tasks not listed under `Dependency-Graph Buildable Tasks` failed synthesis from `action_lib.json`.
See `mctextworld_task_discovery_report.json` for per-task failure reasons.
