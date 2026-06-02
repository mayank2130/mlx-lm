<script lang="ts">
	import { onMount } from 'svelte';
	import { ArrowLeft, LoaderCircle, RefreshCw, ShieldAlert, Wrench } from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { Separator } from '$lib/components/ui/separator';
	import { toolsStore } from '$lib/stores/tools.svelte';
	import { cn } from '$lib/utils';

	onMount(() => {
		toolsStore.initialize();
	});
</script>

<svelte:head>
	<title>Settings · MLX Chat</title>
</svelte:head>

<div class="min-h-dvh bg-background text-foreground">
	<div class="mx-auto flex w-full max-w-3xl flex-col gap-6 px-4 py-8">
		<div class="flex items-center justify-between gap-4">
			<div class="flex items-center gap-3">
				<Button href="/" variant="ghost" size="icon-sm" aria-label="Back to chat">
					<ArrowLeft class="size-4" />
				</Button>
				<div>
					<h1 class="text-2xl font-semibold tracking-tight">Settings</h1>
					<p class="text-sm text-muted-foreground">
						Built-in server tools and chat tool-calling options
					</p>
				</div>
			</div>

			<Button
				variant="outline"
				size="sm"
				onclick={() => void toolsStore.fetchBuiltinTools()}
				disabled={toolsStore.loading}
			>
				{#if toolsStore.loading}
					<LoaderCircle class="size-4 animate-spin" />
				{:else}
					<RefreshCw class="size-4" />
				{/if}
				Refresh
			</Button>
		</div>

		<section class="rounded-2xl border border-border bg-card/40 p-5">
			<div class="flex items-start justify-between gap-4">
				<div>
					<h2 class="text-lg font-medium">Tool calling</h2>
					<p class="mt-1 text-sm text-muted-foreground">
						When enabled, enabled tools below are sent to the model during chat. The model can
						request tool execution and the server will run them.
					</p>
				</div>
				<label class="flex items-center gap-2 text-sm">
					<input
						type="checkbox"
						class="size-4 accent-foreground"
						checked={toolsStore.toolsEnabled}
						onchange={(event) =>
							toolsStore.setToolsEnabled((event.currentTarget as HTMLInputElement).checked)}
					/>
					Enabled
				</label>
			</div>

			{#if toolsStore.isToolsEndpointUnreachable}
				<div
					class="mt-4 flex items-start gap-3 rounded-xl border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-100"
				>
					<ShieldAlert class="mt-0.5 size-4 shrink-0" />
					<div>
						<p class="font-medium">Server tools are unavailable</p>
						<p class="mt-1 text-amber-100/80">
							Start the MLX server with built-in tools enabled, for example:
							<code class="rounded bg-black/30 px-1 py-0.5">mlx_lm.server --model ... --tools</code>
						</p>
					</div>
				</div>
			{:else if toolsStore.error}
				<p class="mt-4 text-sm text-destructive">{toolsStore.error}</p>
			{/if}
		</section>

		<section class="rounded-2xl border border-border bg-card/40 p-5">
			<div class="mb-4 flex items-center gap-2">
				<Wrench class="size-4 text-muted-foreground" />
				<h2 class="text-lg font-medium">Built-in tools</h2>
			</div>

			{#if toolsStore.loading && toolsStore.builtinTools.length === 0}
				<div class="flex items-center gap-2 text-sm text-muted-foreground">
					<LoaderCircle class="size-4 animate-spin" />
					Loading tools...
				</div>
			{:else if toolsStore.builtinTools.length === 0}
				<p class="text-sm text-muted-foreground">No built-in tools were returned by the server.</p>
			{:else}
				<div class="space-y-3">
					{#each toolsStore.builtinTools as tool (tool.function.name)}
						<div class="rounded-xl border border-border/70 bg-background/40 px-4 py-3">
							<div class="flex items-start justify-between gap-4">
								<div class="min-w-0">
									<div class="flex flex-wrap items-center gap-2">
										<h3 class="font-medium">{tool.function.name}</h3>
										<span
											class="rounded-full border border-border px-2 py-0.5 text-[11px] tracking-wide text-muted-foreground uppercase"
										>
											builtin
										</span>
									</div>
									{#if tool.function.description}
										<p class="mt-1 text-sm text-muted-foreground">
											{tool.function.description}
										</p>
									{/if}
								</div>

								<label class="flex shrink-0 items-center gap-2 text-sm">
									<input
										type="checkbox"
										class="size-4 accent-foreground"
										checked={toolsStore.isToolEnabled(tool.function.name)}
										onchange={(event) =>
											toolsStore.setToolEnabled(
												tool.function.name,
												(event.currentTarget as HTMLInputElement).checked
											)}
									/>
									Enabled
								</label>
							</div>

							{#if tool.function.parameters}
								<details class="mt-3">
									<summary
										class={cn(
											'cursor-pointer text-xs text-muted-foreground',
											'hover:text-foreground'
										)}
									>
										Parameters schema
									</summary>
									<pre
										class="mt-2 overflow-x-auto rounded-lg bg-muted/40 p-3 text-xs leading-relaxed text-muted-foreground"
									>{JSON.stringify(tool.function.parameters, null, 2)}</pre>
								</details>
							{/if}
						</div>
					{/each}
				</div>
			{/if}
		</section>

		<Separator />

		<p class="text-sm text-muted-foreground">
			Tool execution happens on the machine running the MLX server. Only enable write-capable tools
			if you trust the model and the environment.
		</p>
	</div>
</div>
