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
		Settings,
		SquarePen,
		Trash2
	} from '@lucide/svelte';
	import { Button } from '$lib/components/ui/button';
	import { Select, SelectContent, SelectItem, SelectTrigger } from '$lib/components/ui/select';
	import {
		ChatPersistence,
		type MessageMeta,
		type PersistedConversation,
		type PersistedMessage
	} from '$lib/stores/database';
	import { toolsStore } from '$lib/stores/tools.svelte';
	import {
		buildApiMessages,
		consumeChatCompletionStream,
		invokeToolCall
	} from '$lib/chat/tool-loop';
	import type { ToolCall } from '$lib/types';
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

	type ChatMessage = Omit<PersistedMessage, 'convId'>;

	const defaultModel = 'default_model';
	const defaultConversationName = 'New chat';

	let models = $state<string[]>([defaultModel]);
	let selectedModel = $state(defaultModel);
	let prompt = $state('');
	let maxTokens = $state(256);
	let temperature = $state(0.2);
	let conversations = $state<PersistedConversation[]>([]);
	let activeConversationId = $state<string | null>(null);
	let messages = $state<ChatMessage[]>([]);
	let isLoadingModels = $state(false);
	let isStreaming = $state(false);
	let streamError = $state('');
	let modelsError = $state('');
	let streamViewport = $state<HTMLElement | null>(null);
	let stickToBottom = $state(true);
	let isAutoScrolling = false;
	const scrollStickThreshold = 80;
	let sidebarExpanded = $state(false);
	let modelPickerOpen = $state(false);
	let modelPickerStyle = $state('');
	let modelTriggerEl = $state<HTMLButtonElement | null>(null);
	let editingMessageId = $state<string | null>(null);
	let copiedMessageId = $state<string | null>(null);
	let hasHydrated = false;
	let copyFeedbackTimer: ReturnType<typeof setTimeout> | null = null;
	const messageActionClass =
		'h-8 w-8 shrink-0 px-0 text-muted-foreground hover:bg-muted/60 hover:text-foreground';
	let persistTimer: ReturnType<typeof setTimeout> | null = null;
	let activeConversation = $derived(
		conversations.find((conversation) => conversation.id === activeConversationId) ?? null
	);
	let chatTitle = $derived(activeConversation?.name || 'MLX Chat');

	onMount(() => {
		toolsStore.initialize();

		void (async () => {
			const storedSettings = ChatPersistence.getUiSettings();
			if (storedSettings.selectedModel) {
				selectedModel = storedSettings.selectedModel;
			}
			if (typeof storedSettings.maxTokens === 'number') {
				maxTokens = storedSettings.maxTokens;
			}
			if (typeof storedSettings.temperature === 'number') {
				temperature = storedSettings.temperature;
			}

			await loadModels();
			await hydrateConversations();
			hasHydrated = true;
		})();

		return () => {
			clearPersistTimer();
			if (copyFeedbackTimer) {
				clearTimeout(copyFeedbackTimer);
			}
		};
	});

	$effect(() => {
		if (messages.length === 0) {
			return;
		}

		const lastMessageContent = messages.at(-1)?.content ?? '';
		if ((lastMessageContent || messages.length > 0) && stickToBottom) {
			void scrollToBottom({ force: true });
		}
	});

	$effect(() => {
		if (!hasHydrated) {
			return;
		}

		ChatPersistence.setUiSettings({
			selectedModel,
			maxTokens,
			temperature
		});

		if (activeConversationId && !isStreaming) {
			schedulePersistActiveConversation();
		}
	});

	$effect(() => {
		if (!hasHydrated) {
			return;
		}

		ChatPersistence.setActiveConversationId(activeConversationId);
	});

	function nextId() {
		return crypto.randomUUID();
	}

	function clearPersistTimer() {
		if (persistTimer) {
			clearTimeout(persistTimer);
			persistTimer = null;
		}
	}

	function schedulePersistActiveConversation() {
		if (!hasHydrated || !activeConversationId) {
			return;
		}

		clearPersistTimer();
		persistTimer = setTimeout(() => {
			void persistActiveConversation();
		}, 120);
	}

	function deriveConversationName(
		messageList: ChatMessage[],
		fallback = defaultConversationName
	): string {
		const firstUserMessage = messageList.find((message) => message.role === 'user')?.content.trim();
		return firstUserMessage ? firstUserMessage.slice(0, 48) : fallback;
	}

	function upsertConversationInState(conversation: PersistedConversation) {
		conversations = [
			conversation,
			...conversations.filter((existing) => existing.id !== conversation.id)
		].sort((left, right) => right.lastModified - left.lastModified);
	}

	async function touchActiveConversation() {
		if (!activeConversationId || !activeConversation) {
			return;
		}

		const updatedConversation: PersistedConversation = {
			...activeConversation,
			name: deriveConversationName(messages, activeConversation.name),
			lastModified: Date.now(),
			model: selectedModel,
			maxTokens,
			temperature
		};

		await ChatPersistence.saveConversation(updatedConversation);
		upsertConversationInState(updatedConversation);
	}

	async function hydrateConversations() {
		conversations = await ChatPersistence.getAllConversations();

		if (conversations.length === 0) {
			activeConversationId = null;
			messages = [];
			return;
		}

		const preferredConversation =
			conversations.find(
				(conversation) => conversation.id === ChatPersistence.getActiveConversationId()
			) ?? conversations[0];

		await openConversation(preferredConversation.id);
	}

	async function ensureActiveConversation() {
		if (activeConversation) {
			return activeConversation;
		}

		const now = Date.now();
		const conversation: PersistedConversation = {
			id: nextId(),
			name: defaultConversationName,
			createdAt: now,
			lastModified: now,
			model: selectedModel,
			maxTokens,
			temperature
		};

		await ChatPersistence.saveConversation(conversation);
		upsertConversationInState(conversation);
		activeConversationId = conversation.id;
		return conversation;
	}

	async function persistActiveConversation() {
		if (!activeConversationId || !activeConversation) {
			return;
		}

		const convId = activeConversationId;
		const now = Date.now();
		const updatedConversation: PersistedConversation = {
			...activeConversation,
			name: deriveConversationName(messages, activeConversation.name),
			lastModified: now,
			model: selectedModel,
			maxTokens,
			temperature
		};

		await ChatPersistence.saveConversation(updatedConversation);
		await ChatPersistence.replaceConversationMessages(
			convId,
			messages.map((message) => ({
				...message,
				convId
			}))
		);

		upsertConversationInState(updatedConversation);
	}

	async function persistMessage(message: ChatMessage) {
		if (!activeConversationId) {
			return;
		}

		await ChatPersistence.saveMessage({
			...message,
			convId: activeConversationId
		});
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
		activeConversationId = null;
		messages = [];
		prompt = '';
		streamError = '';
		editingMessageId = null;
		stickToBottom = true;
	}

	async function openConversation(convId: string) {
		if (isStreaming) {
			return;
		}

		const conversation = conversations.find((entry) => entry.id === convId);
		if (!conversation) {
			return;
		}

		activeConversationId = conversation.id;
		selectedModel = conversation.model || defaultModel;
		maxTokens = conversation.maxTokens || 256;
		temperature = conversation.temperature ?? 0.2;
		messages = (await ChatPersistence.getConversationMessages(convId)).map((message) => ({
			id: message.id,
			role: message.role,
			content: message.content,
			createdAt: message.createdAt,
			meta: message.meta
		}));
		prompt = '';
		streamError = '';
		editingMessageId = null;
		stickToBottom = true;
	}

	async function deleteConversationById(convId: string) {
		if (isStreaming) {
			return;
		}

		await ChatPersistence.deleteConversation(convId);
		const remaining = conversations.filter((conversation) => conversation.id !== convId);
		conversations = remaining;

		if (activeConversationId !== convId) {
			return;
		}

		if (remaining.length === 0) {
			startNewChat();
			return;
		}

		await openConversation(remaining[0].id);
	}

	async function sendPrompt(regenerate = false) {
		const trimmed = prompt.trim();
		if ((!trimmed && !regenerate) || isStreaming) {
			return;
		}

		streamError = '';
		isStreaming = true;
		stickToBottom = true;

		if (!regenerate) {
			await ensureActiveConversation();
		}

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
			schedulePersistActiveConversation();
		} else {
			if (editingMessageId) {
				const editIndex = messages.findIndex((message) => message.id === editingMessageId);
				if (editIndex >= 0) {
					const removedMessages = messages.slice(editIndex);
					if (activeConversationId) {
						for (const message of removedMessages) {
							await ChatPersistence.deleteMessage(message.id);
						}
					}
					messages = messages.slice(0, editIndex);
				}
				editingMessageId = null;
			}

			conversationHistory = messages.map(({ role, content }) => ({ role, content }));
			userMessage = {
				id: nextId(),
				role: 'user',
				content: trimmed,
				createdAt: Date.now()
			};
			messages = [...messages, userMessage];
			prompt = '';
			await persistMessage(userMessage);
		}

		const assistantMessage: ChatMessage = {
			id: nextId(),
			role: 'assistant',
			content: '',
			createdAt: Date.now()
		};

		messages = [...messages, assistantMessage];
		await persistMessage(assistantMessage);
		await touchActiveConversation();
		schedulePersistActiveConversation();
		await scrollToBottom({ force: true });

		const startedAt = performance.now();
		const enabledTools = toolsStore.enabledToolDefinitions;
		let apiMessages = buildApiMessages(messages.slice(0, -1));
		let activeAssistantId = assistantMessage.id;
		const maxToolRounds = 8;

		const appendToActiveAssistant = (fragment: string) => {
			messages = messages.map((message) =>
				message.id === activeAssistantId
					? {
							...message,
							content: `${message.content}${fragment}`
						}
					: message
			);
			if (activeConversationId) {
				const active = messages.find((message) => message.id === activeAssistantId);
				if (active) {
					void ChatPersistence.updateMessage(activeAssistantId, {
						content: active.content
					});
				}
			}
			schedulePersistActiveConversation();
		};

		try {
			for (let round = 0; round < maxToolRounds; round += 1) {
				const response = await fetch('/api/chat', {
					method: 'POST',
					headers: {
						'content-type': 'application/json'
					},
					body: JSON.stringify({
						model: selectedModel,
						messages: apiMessages,
						max_tokens: maxTokens,
						temperature,
						tools: enabledTools.length > 0 ? enabledTools : undefined,
						stream: true
					})
				});

				if (!response.ok || !response.body) {
					const payload = await response.json().catch(() => null);
					throw new Error(payload?.details || payload?.error || 'Streaming request failed');
				}

				const result = await consumeChatCompletionStream(response.body, appendToActiveAssistant);

				const hasToolCalls =
					result.toolCalls.length > 0 || result.finishReason === 'tool_calls';

				if (!hasToolCalls) {
					break;
				}

				updateAssistantToolCalls(activeAssistantId, result.toolCalls);

				apiMessages = [
					...apiMessages,
					{
						role: 'assistant',
						content: result.content,
						tool_calls: result.toolCalls
					}
				];

				for (const toolCall of result.toolCalls) {
					const toolContent = await invokeToolCall(toolCall);
					const toolMessage: ChatMessage = {
						id: nextId(),
						role: 'tool',
						content: toolContent,
						toolName: toolCall.function.name,
						toolCallId: toolCall.id,
						createdAt: Date.now()
					};
					messages = [...messages, toolMessage];
					await persistMessage(toolMessage);
					apiMessages = [
						...apiMessages,
						{
							role: 'tool',
							name: toolCall.function.name,
							tool_call_id: toolCall.id,
							content: toolContent
						}
					];
				}

				if (round + 1 >= maxToolRounds) {
					throw new Error('Tool loop exceeded maximum rounds');
				}

				const nextAssistant: ChatMessage = {
					id: nextId(),
					role: 'assistant',
					content: '',
					createdAt: Date.now()
				};
				activeAssistantId = nextAssistant.id;
				messages = [...messages, nextAssistant];
				await persistMessage(nextAssistant);
				await scrollToBottom({ force: true });
			}

			const durationMs = performance.now() - startedAt;
			const lastAssistant = messages.findLast((message) => message.role === 'assistant');
			const content = lastAssistant?.content ?? '';
			const tokens = estimateTokens(content);
			if (lastAssistant) {
				finalizeAssistantMessage(lastAssistant.id, {
					model: selectedModel,
					tokens,
					durationMs,
					tokensPerSecond: tokens / (durationMs / 1000)
				});
			}
		} catch (error) {
			streamError = error instanceof Error ? error.message : 'Streaming request failed';
			appendToActiveAssistant(`\n\n[error] ${streamError}`);
		} finally {
			isStreaming = false;
			await persistActiveConversation();
			if (stickToBottom) {
				await scrollToBottom({ force: true });
			}
		}
	}

	function updateAssistantToolCalls(messageId: string, toolCalls: ToolCall[]) {
		messages = messages.map((message) =>
			message.id === messageId ? { ...message, toolCalls } : message
		);
		if (activeConversationId) {
			void ChatPersistence.updateMessage(messageId, { toolCalls });
		}
		schedulePersistActiveConversation();
	}

	function finalizeAssistantMessage(messageId: string, meta: MessageMeta) {
		messages = messages.map((message) =>
			message.id === messageId ? { ...message, meta } : message
		);
		if (activeConversationId) {
			void ChatPersistence.updateMessage(messageId, { meta });
		}
		schedulePersistActiveConversation();
	}

	function onChatScroll() {
		if (isAutoScrolling || !streamViewport) {
			return;
		}

		const distanceFromBottom =
			streamViewport.scrollHeight - streamViewport.scrollTop - streamViewport.clientHeight;
		stickToBottom = distanceFromBottom <= scrollStickThreshold;
	}

	async function scrollToBottom({ force = false }: { force?: boolean } = {}) {
		if (!force && !stickToBottom) {
			return;
		}

		await tick();
		if (!streamViewport) {
			return;
		}

		isAutoScrolling = true;
		requestAnimationFrame(() => {
			if (!streamViewport) {
				isAutoScrolling = false;
				return;
			}

			streamViewport.scrollTop = streamViewport.scrollHeight;
			requestAnimationFrame(() => {
				isAutoScrolling = false;
			});
		});
	}

	function onComposerKeydown(event: KeyboardEvent) {
		if (event.key === 'Escape' && editingMessageId) {
			event.preventDefault();
			cancelEdit();
			return;
		}

		if (event.key === 'Enter' && !event.shiftKey) {
			event.preventDefault();
			void sendPrompt();
		}
	}

	function onSubmit(event: SubmitEvent) {
		event.preventDefault();
		void sendPrompt();
	}

	function copyTextToClipboard(text: string): boolean {
		const textarea = document.createElement('textarea');
		textarea.value = text;
		textarea.setAttribute('readonly', '');
		textarea.style.position = 'fixed';
		textarea.style.top = '0';
		textarea.style.left = '0';
		textarea.style.opacity = '0';
		textarea.style.pointerEvents = 'none';
		document.body.appendChild(textarea);
		textarea.focus();
		textarea.select();

		let copied = false;
		try {
			copied = document.execCommand('copy');
		} catch {
			copied = false;
		}

		document.body.removeChild(textarea);

		if (!copied && navigator.clipboard?.writeText) {
			void navigator.clipboard.writeText(text);
			return true;
		}

		return copied;
	}

	function copyMessage(content: string, messageId?: string) {
		const trimmed = content.trim();
		if (!trimmed) {
			return;
		}

		const copied = copyTextToClipboard(trimmed);
		if (!copied || !messageId) {
			return;
		}

		copiedMessageId = messageId;
		if (copyFeedbackTimer) {
			clearTimeout(copyFeedbackTimer);
		}
		copyFeedbackTimer = setTimeout(() => {
			if (copiedMessageId === messageId) {
				copiedMessageId = null;
			}
		}, 1500);
	}

	async function deleteMessage(id: string) {
		if (isStreaming) {
			return;
		}
		const index = messages.findIndex((message) => message.id === id);
		if (index < 0) {
			return;
		}

		if (messages[index].role === 'assistant' && index > 0 && messages[index - 1].role === 'user') {
			if (activeConversationId) {
				await ChatPersistence.deleteMessage(messages[index].id);
				await ChatPersistence.deleteMessage(messages[index - 1].id);
			}
			messages = messages.filter(
				(_, messageIndex) => messageIndex !== index && messageIndex !== index - 1
			);
			await touchActiveConversation();
			schedulePersistActiveConversation();
			return;
		}

		if (activeConversationId) {
			await ChatPersistence.deleteMessage(id);
		}
		messages = messages.filter((message) => message.id !== id);
		if (editingMessageId === id) {
			editingMessageId = null;
			prompt = '';
		}
		await touchActiveConversation();
		schedulePersistActiveConversation();
	}

	function startEditUserMessage(id: string) {
		if (isStreaming) {
			return;
		}

		const message = messages.find((entry) => entry.id === id);
		if (!message || message.role !== 'user') {
			return;
		}

		editingMessageId = id;
		prompt = message.content;
	}

	function cancelEdit() {
		editingMessageId = null;
		prompt = '';
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
</script>

<svelte:head>
	<title>{chatTitle} · MLX Chat</title>
	<meta name="description" content="Local MLX-LM chat" />
</svelte:head>

<div class="flex h-dvh min-h-0 bg-background">
	<aside
		class={cn(
			'flex shrink-0 flex-col border-r border-sidebar-border bg-sidebar transition-[width] duration-200',
			sidebarExpanded ? 'w-52' : 'w-14'
		)}
	>
		<div class="flex flex-col items-center gap-2 p-3">
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

			<Button
				variant="ghost"
				size={sidebarExpanded ? 'default' : 'icon-sm'}
				class={cn(
					'text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground',
					sidebarExpanded && 'w-full justify-start gap-2 px-2'
				)}
				onclick={startNewChat}
				aria-label="New chat"
			>
				<SquarePen class="size-4 shrink-0" />
				{#if sidebarExpanded}
					<span class="truncate text-sm">New chat</span>
				{/if}
			</Button>
		</div>

		<div class="min-h-0 flex-1 overflow-y-auto px-3 pb-3">
			{#if sidebarExpanded}
				<div
					class="px-2 pb-2 text-[11px] font-medium tracking-[0.16em] text-sidebar-foreground/60 uppercase"
				>
					Recent chats
				</div>

				{#if conversations.length === 0}
					<p class="px-2 text-sm text-sidebar-foreground/55">No saved chats yet.</p>
				{:else}
					<div class="space-y-2">
						{#each conversations as conversation (conversation.id)}
							<div class="group flex items-center gap-2">
								<button
									type="button"
									class={cn(
										'min-w-0 flex-1 rounded-xl px-3 py-2 text-left transition-colors',
										activeConversationId === conversation.id
											? 'bg-sidebar-accent text-sidebar-accent-foreground'
											: 'text-sidebar-foreground hover:bg-sidebar-accent/70'
									)}
									onclick={() => void openConversation(conversation.id)}
								>
									<div class="truncate text-sm font-medium">{conversation.name}</div>
									<div class="truncate text-xs text-sidebar-foreground/55">
										{conversation.model}
									</div>
								</button>

								<Button
									variant="ghost"
									size="icon-sm"
									class="shrink-0 text-sidebar-foreground/60 opacity-0 transition-opacity group-hover:opacity-100 hover:bg-sidebar-accent hover:text-sidebar-accent-foreground"
									onclick={() => void deleteConversationById(conversation.id)}
									aria-label={`Delete ${conversation.name}`}
								>
									<Trash2 class="size-4" />
								</Button>
							</div>
						{/each}
					</div>
				{/if}
			{/if}
		</div>

		<div class="mt-auto flex flex-col items-center gap-2 p-3">
			<Button
				variant="ghost"
				size={sidebarExpanded ? 'default' : 'icon-sm'}
				class={cn(
					'text-sidebar-foreground hover:bg-sidebar-accent hover:text-sidebar-accent-foreground',
					sidebarExpanded && 'w-full justify-start gap-2 px-2'
				)}
				href="/settings"
				aria-label="Settings"
			>
				<Settings class="size-4 shrink-0" />
				{#if sidebarExpanded}
					<span class="truncate text-sm">Settings</span>
				{/if}
			</Button>
		</div>
	</aside>

	<main class="relative flex min-h-0 min-w-0 flex-1 flex-col">
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
			<div
				bind:this={streamViewport}
				class="min-h-0 flex-1 overflow-y-auto overscroll-contain"
				onscroll={onChatScroll}
			>
				<div class="mx-auto flex w-full max-w-3xl flex-col gap-8 px-4 py-8 pb-44">
					{#each messages as message (message.id)}
						{#if message.role === 'user'}
							<div class="flex flex-col items-end gap-2">
								<div
									class={cn(
										'max-w-[85%] rounded-3xl bg-muted px-4 py-2.5 text-[15px] leading-relaxed text-foreground',
										editingMessageId === message.id && 'ring-2 ring-ring/50'
									)}
								>
									<p class="break-words whitespace-pre-wrap">{message.content}</p>
								</div>
								<div class="flex items-center gap-1.5 px-1 text-muted-foreground/70">
									<Button
										type="button"
										variant="ghost"
										size="icon-sm"
										class={messageActionClass}
										onclick={() => copyMessage(message.content, message.id)}
										aria-label="Copy message"
									>
										{#if copiedMessageId === message.id}
											<Check class="size-4 text-emerald-400" />
										{:else}
											<Copy class="size-4" />
										{/if}
									</Button>
									<Button
										type="button"
										variant="ghost"
										size="icon-sm"
										class={messageActionClass}
										onclick={() => startEditUserMessage(message.id)}
										aria-label="Edit message"
									>
										<Pencil class="size-4" />
									</Button>
									<Button
										type="button"
										variant="ghost"
										size="icon-sm"
										class={messageActionClass}
										onclick={() => void deleteMessage(message.id)}
										aria-label="Delete message"
									>
										<Trash2 class="size-4" />
									</Button>
								</div>
							</div>
						{:else if message.role === 'tool'}
							<div
								class="rounded-xl border border-border/70 bg-muted/30 px-4 py-3 text-sm text-muted-foreground"
							>
								<p class="mb-2 text-xs font-medium tracking-wide text-foreground uppercase">
									Tool result · {message.toolName}
								</p>
								<pre
									class="max-h-64 overflow-auto whitespace-pre-wrap break-words text-[13px] leading-relaxed"
								>{message.content}</pre>
							</div>
						{:else}
							<div class="flex flex-col gap-3">
								{#if message.toolCalls?.length}
									<div class="flex flex-wrap gap-2">
										{#each message.toolCalls as toolCall (toolCall.id)}
											<span
												class="rounded-full border border-border bg-muted/60 px-2.5 py-1 text-xs text-muted-foreground"
											>
												Calling {toolCall.function.name}
											</span>
										{/each}
									</div>
								{/if}
								<div class="text-[15px] leading-relaxed text-foreground">
									<p class="break-words whitespace-pre-wrap">
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

								<div class="flex items-center gap-1.5 px-1">
									<Button
										type="button"
										variant="ghost"
										size="icon-sm"
										class={messageActionClass}
										onclick={() => copyMessage(message.content, message.id)}
										aria-label="Copy response"
									>
										{#if copiedMessageId === message.id}
											<Check class="size-4 text-emerald-400" />
										{:else}
											<Copy class="size-4" />
										{/if}
									</Button>
									<Button
										type="button"
										variant="ghost"
										size="icon-sm"
										class={messageActionClass}
										disabled={isStreaming}
										onclick={() => void sendPrompt(true)}
										aria-label="Regenerate response"
									>
										<RefreshCw class="size-4" />
									</Button>
									<Button
										type="button"
										variant="ghost"
										size="icon-sm"
										class={messageActionClass}
										onclick={() => void deleteMessage(message.id)}
										aria-label="Delete response"
									>
										<Trash2 class="size-4" />
									</Button>
								</div>
							</div>
						{/if}
					{/each}
				</div>
			</div>
		{/if}

		{#if messages.length > 0}
			<div class="pointer-events-none absolute inset-x-0 bottom-0 z-10 px-4 pt-4 pb-6">
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
			{#if editingMessageId}
				<div
					class="flex items-center justify-between gap-3 border-b border-border/60 px-5 py-2.5 text-sm text-muted-foreground"
				>
					<span>Editing your message — send to replace, or cancel to keep it.</span>
					<Button type="button" variant="ghost" size="sm" class="shrink-0" onclick={cancelEdit}>
						Cancel
					</Button>
				</div>
			{/if}
			<textarea
				bind:value={prompt}
				rows={1}
				class={cn(
					'max-h-40 min-h-[52px] w-full resize-none bg-transparent px-5 text-[15px] leading-relaxed text-foreground outline-none placeholder:text-muted-foreground',
					editingMessageId ? 'pt-3' : 'pt-4'
				)}
				placeholder={editingMessageId ? 'Edit your message...' : 'Type a message...'}
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
					<div
						class="flex items-center gap-2 rounded-full border border-border/80 bg-muted/40 px-2.5 py-1 text-xs text-muted-foreground"
					>
						<label class="whitespace-nowrap" for="composer-max-tokens">Max</label>
						<input
							id="composer-max-tokens"
							type="number"
							min="1"
							max="4096"
							step="1"
							bind:value={maxTokens}
							class="w-16 border-0 bg-transparent p-0 text-right text-foreground outline-none"
						/>
					</div>

					<div
						class="flex items-center gap-2 rounded-full border border-border/80 bg-muted/40 px-2.5 py-1 text-xs text-muted-foreground"
					>
						<label class="whitespace-nowrap" for="composer-temperature">Temp</label>
						<input
							id="composer-temperature"
							type="number"
							min="0"
							max="2"
							step="0.1"
							bind:value={temperature}
							class="w-12 border-0 bg-transparent p-0 text-right text-foreground outline-none"
						/>
					</div>

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
