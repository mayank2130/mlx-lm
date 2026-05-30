import { env } from '$env/dynamic/private';
import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

const DEFAULT_MLX_BASE_URL = 'http://127.0.0.1:8080';

function getMlxBaseUrl() {
	return env.MLX_BASE_URL || DEFAULT_MLX_BASE_URL;
}

export const POST: RequestHandler = async ({ fetch, request }) => {
	const body = await request.json();
	const upstream = await fetch(`${getMlxBaseUrl()}/v1/chat/completions`, {
		method: 'POST',
		headers: {
			'content-type': 'application/json'
		},
		body: JSON.stringify({
			stream: true,
			...body
		})
	});

	if (!upstream.ok || !upstream.body) {
		const message = await upstream.text();
		return json(
			{
				error: 'Failed to start chat completion',
				details: message
			},
			{ status: upstream.status || 502 }
		);
	}

	return new Response(upstream.body, {
		status: upstream.status,
		headers: {
			'content-type': 'text/event-stream',
			'cache-control': 'no-cache'
		}
	});
};
