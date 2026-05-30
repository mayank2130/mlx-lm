import { env } from '$env/dynamic/private';
import { json } from '@sveltejs/kit';
import type { RequestHandler } from './$types';

const DEFAULT_MLX_BASE_URL = 'http://127.0.0.1:8080';

function getMlxBaseUrl() {
	return env.MLX_BASE_URL || DEFAULT_MLX_BASE_URL;
}

export const GET: RequestHandler = async ({ fetch }) => {
	const upstream = await fetch(`${getMlxBaseUrl()}/v1/models`);

	if (!upstream.ok) {
		const message = await upstream.text();
		return json(
			{
				error: 'Failed to load models from MLX server',
				details: message
			},
			{ status: upstream.status }
		);
	}

	const payload = await upstream.json();
	return json(payload);
};
