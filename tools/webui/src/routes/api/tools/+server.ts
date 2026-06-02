import { env } from '$env/dynamic/private';
import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

const DEFAULT_MLX_BASE_URL = 'http://127.0.0.1:8080';

function getMlxBaseUrl() {
	return env.MLX_BASE_URL || DEFAULT_MLX_BASE_URL;
}

export const GET: RequestHandler = async ({ fetch }) => {
	const upstream = await fetch(`${getMlxBaseUrl()}/tools`);

	if (!upstream.ok) {
		const message = await upstream.text();
		return json(
			{
				error: 'Failed to list tools',
				details: message
			},
			{ status: upstream.status || 502 }
		);
	}

	const payload = await upstream.json();
	return json(payload);
};

export const POST: RequestHandler = async ({ fetch, request }) => {
	const body = await request.json();
	const upstream = await fetch(`${getMlxBaseUrl()}/tools`, {
		method: 'POST',
		headers: {
			'content-type': 'application/json'
		},
		body: JSON.stringify(body)
	});

	const payload = await upstream.json().catch(async () => ({
		error: await upstream.text()
	}));

	if (!upstream.ok) {
		return json(payload, { status: upstream.status || 502 });
	}

	return json(payload);
};
