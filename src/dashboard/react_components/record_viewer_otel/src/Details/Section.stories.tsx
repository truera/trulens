import type { Meta, StoryObj } from '@storybook/react';
import Section from './Section';
import { Box, Typography } from '@mui/material';

type Story = StoryObj<typeof Section>;

const meta: Meta<typeof Section> = {
  title: 'Components/Section',
  component: Section,
  args: {
    title: 'Basic Section',
    body: 'This is a basic section with title, subtitle and body text.',
  },
};

export default meta;

export const Default: Story = {};

export const WithChildren: Story = {
  args: {
    title: 'Section With Children',
    body: 'This section contains child components.',
    children: (
      <Box sx={{ mt: 2, p: 2, bgcolor: 'background.paper', borderRadius: 1 }}>
        <Typography variant="body2">This is a child component inside the section.</Typography>
      </Box>
    ),
  },
};

export const LongContent: Story = {
  args: {
    title: 'Section With Long Content',
    body: 'Lorem ipsum dolor sit amet, consectetur adipiscing elit. Nullam euismod, nisi vel consectetur interdum, nisl nisi consectetur nisi, eget consectetur nisi nisi vel nisi. Nullam euismod, nisi vel consectetur interdum, nisl nisi consectetur nisi, eget consectetur nisi nisi vel nisi.',
  },
};

export const WithoutBody: Story = {
  args: {
    title: 'Section Without Body',
    children: (
      <Typography variant="body2" color="text.secondary">
        This section has no body text, only children.
      </Typography>
    ),
  },
};
