from dataclasses import dataclass

from pydantic_graph.beta import GraphBuilder, StepContext
from pydantic_graph.beta.join import reduce_list_extend
import asyncio

@dataclass
class SimpleState:
    pass

async def main():
    g = GraphBuilder(state_type=SimpleState, output_type=list[int])

    @g.step
    async def source(ctx: StepContext[SimpleState, None, None]) -> int:
        return 10

    @g.step
    async def add_one(ctx: StepContext[SimpleState, None, int]) -> list[int]:
        return [ctx.inputs + 1]

    @g.step
    async def add_two(ctx: StepContext[SimpleState, None, int]) -> list[int]:
        return [ctx.inputs + 2]

    @g.step
    async def add_three(ctx: StepContext[SimpleState, None, int]) -> list[int]:
        return [ctx.inputs + 3]

    collect = g.join(reduce_list_extend, initial_factory=list[int])
    broadcast = g.join(reduce_list_extend, initial_factory=list[int])

    # Broadcasting: send the value from source to all three steps
    g.add(
        g.edge_from(g.start_node).to(source),
        g.edge_from(source).to(add_one, add_two, add_three),
        g.edge_from(add_one, add_two).to(broadcast),
        g.edge_from(broadcast).to(collect),
        g.edge_from(add_three).to(collect),
        g.edge_from(collect).to(g.end_node),
    )

    graph = g.build()
    result = await graph.run(state=SimpleState())
    print(sorted(result))
    #> [11, 12, 13]

if __name__ == "__main__":
    asyncio.run(main())