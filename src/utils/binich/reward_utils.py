

class ShortTermReward:
    def __init__(self):
        self.r_kill = 0
        self.r_fall = 0
        self.r_inferior = 0
        self.r_dominant = 0
        self.r_self_health_loss_ratio = 0
        self.r_enemy_health_loss_ratio = 0
        self.r_fire_coverage = 0
        self.r_covered_in_fire = 0

    def __str__(self):
        return '{} {} {} {} {} {} {} {}' \
            .format(self.r_kill, self.r_fall,
                    self.r_inferior, self.r_dominant,
                    self.r_self_health_loss_ratio, self.r_enemy_health_loss_ratio,
                    self.r_fire_coverage, self.r_covered_in_fire)

    def short_reward(self, agents, enemies, previous_ally_units, previous_enemy_units, shoot_range):
        pre_enemy_units_alive = [enemy for enemy in previous_enemy_units.values() if enemy.health > 0]
        cur_enemy_units_alive = [enemy for enemy in enemies.values() if enemy.health > 0]
        pre_ally_units_alive = [agent for agent in previous_ally_units.values() if agent.health > 0]
        cur_ally_units_alive = [agent for agent in agents.values() if agent.health > 0]
        self.r_kill = (len(pre_enemy_units_alive) - len(cur_enemy_units_alive)) * 10
        self.r_fall = (len(pre_ally_units_alive) - len(cur_ally_units_alive)) * -10

        pre_health_ally = sum([agent.health for agent in previous_ally_units.values()])
        cur_health_ally = sum([agent.health for agent in agents.values()])
        max_health_ally = sum([agent.health_max for agent in previous_ally_units.values()])
        pre_health_enemy = sum([enemy.health for enemy in previous_enemy_units.values()])
        cur_health_enemy = sum([enemy.health for enemy in enemies.values()])
        max_health_enemy = sum([enemy.health_max for enemy in previous_enemy_units.values()])

        if pre_health_ally < pre_health_enemy and (pre_health_ally - cur_health_ally) < (pre_health_enemy - cur_health_enemy):
            self.r_inferior = 10
        if pre_health_ally > pre_health_enemy and (pre_health_ally - cur_health_ally) > (pre_health_enemy - cur_health_enemy):
            self.r_dominant = -10

        self.r_self_health_loss_ratio = (pre_health_ally - cur_health_ally) / max_health_ally * -10
        self.r_enemy_health_loss_ratio = (pre_health_enemy - cur_health_enemy) / max_health_enemy * 10

        max_pre_enemy_attacks = 0
        max_pre_ally_attacks = 0

        for enemy in pre_enemy_units_alive:
            enemy_attacks = 0
            for ally in pre_ally_units_alive:
                distance = ((enemy.pos.x - ally.pos.x) ** 2 + (enemy.pos.y - ally.pos.y) ** 2) ** 0.5
                if distance <= shoot_range:
                    enemy_attacks += 1
            max_pre_enemy_attacks = max(max_pre_enemy_attacks, enemy_attacks)
        for ally in pre_ally_units_alive:
            ally_attacks = 0
            for enemy in pre_enemy_units_alive:
                distance = ((ally.pos.x - enemy.pos.x) ** 2 + (ally.pos.y - enemy.pos.y) ** 2) ** 0.5
                if distance <= shoot_range:
                    ally_attacks += 1
            max_pre_ally_attacks = max(max_pre_ally_attacks, ally_attacks)

        max_cur_enemy_attacks = 0
        max_cur_ally_attacks = 0

        for enemy in cur_enemy_units_alive:
            enemy_attacks = 0
            for ally in cur_ally_units_alive:
                distance = ((enemy.pos.x - ally.pos.x) ** 2 + (enemy.pos.y - ally.pos.y) ** 2) ** 0.5
                if distance <= shoot_range:
                    enemy_attacks += 1
            max_cur_enemy_attacks = max(max_cur_enemy_attacks, enemy_attacks)
        for ally in cur_ally_units_alive:
            ally_attacks = 0
            for enemy in cur_enemy_units_alive:
                distance = ((ally.pos.x - enemy.pos.x) ** 2 + (ally.pos.y - enemy.pos.y) ** 2) ** 0.5
                if distance <= shoot_range:
                    ally_attacks += 1
            max_cur_ally_attacks = max(max_cur_ally_attacks, ally_attacks)

        if max_cur_enemy_attacks > max_pre_enemy_attacks:
            self.r_fire_coverage = (max_cur_enemy_attacks - max_pre_enemy_attacks) * 5
        if max_cur_ally_attacks < max_pre_ally_attacks:
            self.r_covered_in_fire = (max_pre_ally_attacks - max_cur_ally_attacks) * 5
        return (self.r_kill + self.r_fall +
                self.r_inferior + self.r_dominant +
                self.r_self_health_loss_ratio + self.r_enemy_health_loss_ratio +
                self.r_fire_coverage + self.r_covered_in_fire)

