<script lang="ts">
	import { onMount, tick } from 'svelte';
	import {
		ArrowUp,
		Box,
		Check,
		ChevronDown,
		Copy,
		LoaderCircle,
		PanelLeft,
		PanelLeftClose,
		Pencil,
		Plus,
		RefreshCw,
		Search,
		Settings,
		Sparkles,
		SquarePen,
		Trash2
	} from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import {
		Dialog,
		DialogContent,
		DialogDescription,
		DialogFooter,
		DialogHeader,
		DialogTitle
	} from '$lib/components/ui/dialog';
	import { Input } from '$lib/components/ui/input';
	import { ScrollArea } from '$lib/components/ui/scroll-area';
	import {
		Select,
		SelectContent,
		SelectItem,
		SelectTrigger
	} from '$lib/components/ui/select';
	import { Separator } from '$lib/components/ui/separator';
	import { cn } from '$lib/utils';

	type ModelRecord = {
		id: string;
		object?: string;
		created?: number;
	};

	type ModelsResponse = {
		object?: string;
		data?: ModelRecord[];
		error?: string;
		details?: string;
	};

	type MessageMeta = {
		model: string;
		tokens: number;
		durationMs: number;
		tokensPerSecond: number;
	};

	type ChatMessage = {
		id: string;
		role: 'user' | 'assistant' | 'system';
		content: string;
		meta?: MessageMeta;
	};

	const defaultModel = 'default_model';

	let models = $state<string[]>([defaultModel]);
	let selectedModel = $state(defaultModel);
	let prompt = $state('');
	let maxTokens = $state(256);
	let temperature = $state(0.2);
	let messages = $state<ChatMessage[]>([]);
	let isLoadingModels = $state(false);
	let isStreaming = $state(false);
	let streamError = $state('');
	let modelsError = $state('');
	let streamViewport = $state<HTMLElement | null>(null);
	let sidebarExpanded = $state(false);
	let settingsOpen = $state(false);
	let modelPickerOpen = $state(false);
	let modelPickerStyle = $state('');
	let modelTriggerEl = $state<HTMLButtonElement | null>(null);
	let chatTitle = $derived(
		messages.find((message) => message.role === 'user')?.content.slice(0, 48) || 'MLX Chat'
	);

	onMount(async () => {
		await loadModels();
	});

	function nextId() {
		return crypto.randomUUID();
	}

	function estimateTokens(text: string) {
		return Math.max(1, Math.round(text.trim().split(/\s+/).filter(Boolean).length * 1.3));
	}

	async function loadModels() {
		isLoadingModels = true;
		modelsError = '';

		try {
			const response = await fetch('/api/models');
			const payload = (await response.json()) as ModelsResponse;

			if (!response.ok) {
				throw new Error(payload.details || payload.error || 'Unable to load models');
			}

			models = [defaultModel, ...(payload.data || []).map((entry) => entry.id)].filter(
				(model, index, list) => list.indexOf(model) === index
			);

			if (!models.includes(selectedModel)) {
				selectedModel = defaultModel;
			}
		} catch (error) {
			modelsError = error instanceof Error ? error.message : 'Unable to load models';
		} finally {
			isLoadingModels = false;
		}
	}

	function startNewChat() {
		if (isStreaming) {
			return;
		}
		messages = [];
		prompt = '';
		streamError = '';
	}

	async function sendPrompt(regenerate = false) {
		const trimmed = prompt.trim();
		if ((!trimmed && !regenerate) || isStreaming) {
			return;
		}

		streamError = '';
		isStreaming = true;

		let conversationHistory: { role: ChatMessage['role']; content: string }[];
		let userMessage: ChatMessage;

		if (regenerate) {
			const lastUserIndex = [...messages].reverse().findIndex((message) => message.role === 'user');
			if (lastUserIndex < 0) {
				isStreaming = false;
				return;
			}

			const absoluteIndex = messages.length - 1 - lastUserIndex;
			userMessage = messages[absoluteIndex];
			messages = messages.slice(0, absoluteIndex);
			conversationHistory = messages.map(({ role, content }) => ({ role, content }));
		} else {
			conversationHistory = messages.map(({ role, content }) => ({ role, content }));
			userMessage = {
				id: nextId(),
				role: 'user',
				content: trimmed
			};
			messages = [...messages, userMessage];
			prompt = '';
		}

		const assistantMessage: ChatMessage = {
			id: nextId(),
			role: 'assistant',
			content: ''
		};

		messages = [...messages, assistantMessage];
		await scrollToBottom();

		const startedAt = performance.now();

		try {
			const response = await fetch('/api/chat', {
				method: 'POST',
				headers: {
					'content-type': 'application/json'
				},
				body: JSON.stringify({
					model: selectedModel,
					messages: [...conversationHistory, { role: userMessage.role, content: userMessage.content }],
					max_tokens: maxTokens,
					temperature
				})
			});

			if (!response.ok || !response.body) {
				const payload = await response.json().catch(() => null);
				throw new Error(payload?.details || payload?.error || 'Streaming request failed');
			}

			await consumeEventStream(response.body);

			const durationMs = performance.now() - startedAt;
			const content = messages.at(-1)?.content ?? '';
			const tokens = estimateTokens(content);
			finalizeLastAssistantMessage({
				model: selectedModel,
				tokens,
				durationMs,
				tokensPerSecond: tokens / (durationMs / 1000)
			});
		} catch (error) {
			streamError = error instanceof Error ? error.message : 'Streaming request failed';
			updateLastAssistantMessage(`\n\n[error] ${streamError}`);
		} finally {
			isStreaming = false;
			await scrollToBottom();
		}
	}

	async function consumeEventStream(stream: ReadableStream<Uint8Array>) {
		const reader = stream.getReader();
		const decoder = new TextDecoder();
		let buffer = '';

		while (true) {
			const { value, done } = await reader.read();
			if (done) {
				break;
			}

			buffer += decoder.decode(value, { stream: true });
			const events = buffer.split('\n\n');
			buffer = events.pop() || '';

			for (const event of events) {
				for (const line of event.split('\n')) {
					if (!line.startsWith('data: ')) {
						continue;
					}

					const data = line.slice(6).trim();
					if (data === '[DONE]') {
						return;
					}

					const payload = JSON.parse(data);
					const delta = payload?.choices?.[0]?.delta;
					const content = delta?.content;

					if (typeof content === 'string' && content.length > 0) {
						updateLastAssistantMessage(content);
						await scrollToBottom();
					}
				}
			}
		}
	}

	function updateLastAssistantMessage(fragment: string) {
		const lastIndex = messages.length - 1;
		if (lastIndex < 0 || messages[lastIndex]?.role !== 'assistant') {
			return;
		}

		messages = messages.map((message, index) =>
			index === lastIndex
				? {
						...message,
						content: `${message.content}${fragment}`
					}
				: message
		);
	}

	function finalizeLastAssistantMessage(meta: MessageMeta) {
		const lastIndex = messages.length - 1;
		if (lastIndex < 0 || messages[lastIndex]?.role !== 'assistant') {
			return;
		}

		messages = messages.map((message, index) =>
			index === lastIndex ? { ...message, meta } : message
		);
	}

	async function scrollToBottom() {
		await tick();
		if (!streamViewport) {
			return;
		}
		streamViewport.scrollTop = streamViewport.scrollHeight;
	}

	function onComposerKeydown(event: KeyboardEvent) {
		if (event.key === 'Enter' && !event.shiftKey) {
			event.preventDefault();
			void sendPrompt();
		}
	}

	function onSubmit(event: SubmitEvent) {
		event.preventDefault();
		void sendPrompt();
	}

	async function copyMessage(content: string) {
		try {
			await navigator.clipboard.writeText(content);
		} catch {
			// ignore clipboard failures
		}
	}

	function deleteMessage(id: string) {
		if (isStreaming) {
			return;
		}
		const index = messages.findIndex((message) => message.id === id);
		if (index < 0) {
			return;
		}

		if (messages[index].role === 'assistant' && index > 0 && messages[index - 1].role === 'user') {
			messages = messages.filter((_, messageIndex) => messageIndex !== index && messageIndex !== index - 1);
			return;
		}

		messages = messages.filter((message) => message.id !== id);
	}

	function editUserMessage(id: string) {
		const message = messages.find((entry) => entry.id === id);
		if (!message || message.role !== 'user') {
			return;
		}
		prompt = message.content;
		deleteMessage(id);
	}

	function formatDuration(ms: number) {
		return `${(ms / 1000).toFixed(1)}s`;
	}

	function formatSpeed(meta: MessageMeta) {
		return `${meta.tokensPerSecond.toFixed(2)} t/s`;
	}

	function closeModelPicker() {
		modelPickerOpen = false;
	}

	function toggleModelPicker() {
		if (modelPickerOpen) {
			closeModelPicker();
			return;
		}

		if (!modelTriggerEl) {
			return;
		}

		const rect = modelTriggerEl.getBoundingClientRect();
		const spaceAbove = rect.top - 12;
		const maxHeight = Math.min(280, Math.max(120, spaceAbove));
		const width = Math.max(rect.width, 220);

		modelPickerStyle = [
			`position:fixed`,
			`left:${Math.max(12, Math.min(rect.left, window.innerWidth - width - 12))}px`,
			`bottom:${window.innerHeight - rect.top + 8}px`,
			`width:${width}px`,
			`max-height:${maxHeight}px`
		].join(';');

		modelPickerOpen = true;
	}

	function selectModel(model: string) {
		selectedModel = model;
		closeModelPicker();
	}

	const sidebarItems = [
		{ icon: SquarePen, label: 'New chat', action: startNewChat },
		{ icon: Search, label: 'Search', action: () => {} },
		{ icon: Sparkles, label: 'Discover', action: () => {} }
	] as const;
</script>

<svelte:head>
	<title>{chatTitle} · MLX Chat</title>
	<meta name="description" content="Local MLX-LM chat" />
</svelte:head>

<div class="flex h-dvh bg-background">
	<aside
		class={cn(
			'flex shrink-0 flex-col border-r border-sidebar-border bg-sidebar transition-[width] duration-200',
			sidebarExpanded ? 'w-52' : 'w-14'
		)}
	>
		<div class="flex flex-col items-center gap-1 p-2">
			<Button
				variant="ghost"
				size="icon-sm"
				class="text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
				onclick={() => (sidebarExpanded = !sidebarExpanded)}
				aria-label={sidebarExpanded ? 'Collapse sidebar' : 'Expand sidebar'}
			>
				{#if sidebarExpanded}
					<PanelLeftClose class="size-4" />
				{:else}
					<PanelLeft class="size-4" />
				{/if}
			</Button>

			{#each sidebarItems as item (item.label)}
				<Button
					variant="ghost"
					size={sidebarExpanded ? 'default' : 'icon-sm'}
					class={cn(
						'text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground',
						sidebarExpanded && 'w-full justify-start gap-2 px-2'
					)}
					onclick={item.action}
					aria-label={item.label}
				>
					<item.icon class="size-4 shrink-0" />
					{#if sidebarExpanded}
						<span class="truncate text-sm">{item.label}</span>
					{/if}
				</Button>
			{/each}
		</div>

		<div class="mt-auto flex flex-col items-center gap-1 p-2">
			<Button
				variant="ghost"
				size={sidebarExpanded ? 'default' : 'icon-sm'}
				class={cn(
					'text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground',
					sidebarExpanded && 'w-full justify-start gap-2 px-2'
				)}
				onclick={() => (settingsOpen = true)}
				aria-label="Settings"
			>
				<Settings class="size-4 shrink-0" />
				{#if sidebarExpanded}
					<span class="truncate text-sm">Settings</span>
				{/if}
			</Button>
		</div>
	</aside>

	<main class="relative flex min-w-0 flex-1 flex-col">
		{#if messages.length === 0}
			<div class="flex flex-1 flex-col items-center justify-center px-4">
				<div class="flex w-full max-w-3xl flex-col items-center gap-7">
					<div class="text-center">
						<h1 class="text-4xl font-semibold tracking-tight text-foreground">Hello there</h1>
						<p class="mt-2.5 text-[1.05rem] leading-relaxed text-muted-foreground">
							Type a message or upload files to get started
						</p>
					</div>
					<div class="pointer-events-auto w-full">
						{@render composer()}
					</div>
				</div>
			</div>
		{:else}
			<ScrollArea bind:viewportRef={streamViewport} class="min-h-0 flex-1">
				<div class="mx-auto flex w-full max-w-3xl flex-col gap-8 px-4 py-8 pb-36">
					{#each messages as message (message.id)}
						{#if message.role === 'user'}
							<div class="flex flex-col items-end gap-2">
								<div
									class="max-w-[85%] rounded-3xl bg-muted px-4 py-2.5 text-[15px] leading-relaxed text-foreground"
								>
									<p class="whitespace-pre-wrap break-words">{message.content}</p>
								</div>
								<div class="flex items-center gap-0.5 text-muted-foreground/70">
									<Button
										variant="ghost"
										size="icon-xs"
										class="text-muted-foreground"
										onclick={() => copyMessage(message.content)}
										aria-label="Copy"
									>
										<Copy class="size-3.5" />
									</Button>
									<Button
										variant="ghost"
										size="icon-xs"
										class="text-muted-foreground"
										onclick={() => editUserMessage(message.id)}
										aria-label="Edit"
									>
										<Pencil class="size-3.5" />
									</Button>
									<Button
										variant="ghost"
										size="icon-xs"
										class="text-muted-foreground"
										onclick={() => deleteMessage(message.id)}
										aria-label="Delete"
									>
										<Trash2 class="size-3.5" />
									</Button>
								</div>
							</div>
						{:else}
							<div class="flex flex-col gap-3">
								<div class="text-[15px] leading-relaxed text-foreground">
									<p class="whitespace-pre-wrap break-words">
										{message.content || (isStreaming ? '…' : '')}
									</p>
								</div>

								{#if message.meta}
									<div class="flex flex-wrap items-center gap-2 text-xs text-muted-foreground">
										<span
											class="inline-flex items-center gap-1.5 rounded-full border border-border bg-muted/60 px-2.5 py-1"
										>
											<Box class="size-3" />
											{message.meta.model}
										</span>
										<span>{message.meta.tokens} tokens</span>
										<span>{formatDuration(message.meta.durationMs)}</span>
										<span>{formatSpeed(message.meta)}</span>
									</div>
								{/if}

								<div class="flex items-center gap-0.5">
									<Button
										variant="ghost"
										size="icon-xs"
										class="text-muted-foreground"
										onclick={() => copyMessage(message.content)}
										aria-label="Copy"
									>
										<Copy class="size-3.5" />
									</Button>
									<Button
										variant="ghost"
										size="icon-xs"
										class="text-muted-foreground"
										disabled={isStreaming}
										onclick={() => void sendPrompt(true)}
										aria-label="Regenerate"
									>
										<RefreshCw class="size-3.5" />
									</Button>
									<Button
										variant="ghost"
										size="icon-xs"
										class="text-muted-foreground"
										onclick={() => deleteMessage(message.id)}
										aria-label="Delete"
									>
										<Trash2 class="size-3.5" />
									</Button>
								</div>
							</div>
						{/if}
					{/each}
				</div>
			</ScrollArea>
		{/if}

		{#if messages.length > 0}
			<div class="pointer-events-none absolute inset-x-0 bottom-0 z-10 px-4 pb-6 pt-4">
				<div class="pointer-events-auto mx-auto w-full max-w-3xl">
					{@render composer()}
				</div>
			</div>
		{/if}
	</main>
</div>

{#snippet composer()}
	<form class="w-full" onsubmit={onSubmit}>
		<div
			class="overflow-visible rounded-3xl border border-border bg-[oklch(0.2_0_0)] shadow-lg shadow-black/20"
		>
			<textarea
				bind:value={prompt}
				rows={1}
				class="max-h-40 min-h-[52px] w-full resize-none bg-transparent px-5 pt-4 text-[15px] leading-relaxed text-foreground outline-none placeholder:text-muted-foreground"
				placeholder="Type a message..."
				onkeydown={onComposerKeydown}
			></textarea>

			<div class="flex items-center justify-between gap-2 px-3 pb-3">
				<Button
					type="button"
					variant="ghost"
					size="icon-sm"
					class="rounded-full text-muted-foreground"
					disabled
					aria-label="Attach files"
				>
					<Plus class="size-4" />
				</Button>

				<div class="flex items-center gap-2">
					<Button
						bind:ref={modelTriggerEl}
						type="button"
						variant="outline"
						class="h-8 max-w-[220px] gap-1.5 rounded-full border-border/80 bg-muted/50 px-3 text-xs shadow-none"
						onclick={toggleModelPicker}
						aria-expanded={modelPickerOpen}
						aria-haspopup="listbox"
						aria-label="Select model"
					>
						<Box class="size-3.5 shrink-0 text-muted-foreground" />
						<span class="truncate">{selectedModel}</span>
						<ChevronDown
							class={cn(
								'size-3.5 shrink-0 text-muted-foreground transition-transform',
								modelPickerOpen && 'rotate-180'
							)}
						/>
					</Button>

					<Button
						type="submit"
						size="icon-sm"
						class="rounded-full"
						disabled={isStreaming || !prompt.trim()}
						aria-label="Send message"
					>
						{#if isStreaming}
							<LoaderCircle class="size-4 animate-spin" />
						{:else}
							<ArrowUp class="size-4" />
						{/if}
					</Button>
				</div>
			</div>
		</div>

		{#if streamError}
			<p class="mt-2 text-center text-sm text-destructive">{streamError}</p>
		{/if}
		{#if modelsError}
			<p class="mt-2 text-center text-sm text-destructive">{modelsError}</p>
		{/if}
	</form>
{/snippet}

{#if modelPickerOpen}
	<button
		type="button"
		class="fixed inset-0 z-[60] cursor-default bg-transparent"
		aria-label="Close model menu"
		onclick={closeModelPicker}
	></button>
	<ul
		role="listbox"
		aria-label="Models"
		class="z-[70] overflow-y-auto rounded-xl border border-border bg-popover p-1 text-popover-foreground shadow-xl ring-1 ring-foreground/10"
		style={modelPickerStyle}
	>
		{#each models as model (model)}
			<li role="presentation">
				<button
					type="button"
					role="option"
					aria-selected={selectedModel === model}
					class={cn(
						'flex w-full items-center gap-2 rounded-lg px-3 py-2 text-left text-sm hover:bg-accent',
						selectedModel === model && 'bg-accent'
					)}
					onclick={() => selectModel(model)}
				>
					<span class="min-w-0 flex-1 truncate">{model}</span>
					{#if selectedModel === model}
						<Check class="size-4 shrink-0 text-muted-foreground" />
					{/if}
				</button>
			</li>
		{/each}
	</ul>
{/if}

<Dialog bind:open={settingsOpen}>
	<DialogContent class="border-border/80 bg-card sm:max-w-md">
		<DialogHeader>
			<DialogTitle>Settings</DialogTitle>
			<DialogDescription>Generation options for the MLX server proxy.</DialogDescription>
		</DialogHeader>

		<div class="space-y-4 py-2">
			<div class="space-y-2">
				<label class="text-sm font-medium" for="settings-model">Default model</label>
				<Select type="single" bind:value={selectedModel}>
					<SelectTrigger id="settings-model" class="w-full">
						<span class="truncate">{selectedModel}</span>
					</SelectTrigger>
					<SelectContent class="max-h-[300px]">
						{#each models as model (model)}
							<SelectItem value={model} label={model}>{model}</SelectItem>
						{/each}
					</SelectContent>
				</Select>
			</div>

			<div class="grid grid-cols-2 gap-3">
				<div class="space-y-2">
					<label class="text-sm font-medium" for="settings-temperature">Temperature</label>
					<Input
						id="settings-temperature"
						type="number"
						min="0"
						max="2"
						step="0.1"
						bind:value={temperature}
					/>
				</div>
				<div class="space-y-2">
					<label class="text-sm font-medium" for="settings-max-tokens">Max tokens</label>
					<Input
						id="settings-max-tokens"
						type="number"
						min="1"
						max="4096"
						step="1"
						bind:value={maxTokens}
					/>
				</div>
			</div>

			<Separator />

			<div class="space-y-1 text-sm text-muted-foreground">
				<p><span class="font-medium text-foreground">GET</span> /api/models</p>
				<p><span class="font-medium text-foreground">POST</span> /api/chat</p>
			</div>
		</div>

		<DialogFooter>
			<Button variant="outline" onclick={loadModels} disabled={isLoadingModels}>
				{#if isLoadingModels}
					<LoaderCircle class="size-4 animate-spin" />
				{/if}
				Refresh models
			</Button>
		</DialogFooter>
	</DialogContent>
</Dialog>
